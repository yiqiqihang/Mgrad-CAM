from typing import List

import cv2
import numpy as np
import torch

from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class AlignGradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None,
                 alignement_power: float = 1.0,
                 signed: bool = True,         # 是否做“有符号”融合（保留抑制证据）
                 relu_at_end: bool = True):     # 是否在末端做 ReLU（获得传统外观）
        super(
            AlignGradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform,
            uses_gradients=True,
            alignement_power=alignement_power)
        self.signed = signed
        self.relu_at_end = relu_at_end

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        # target_layer: torch.nn.Module,
                        target_layer: List[torch.nn.Module],
                        targers: List[torch.nn.Module],
                        # target_category,
                        activations: torch.Tensor,
                        grads: torch.Tensor, ) -> np.ndarray:
        """
        计算融合了余弦对齐权重的 CAM 通道权重：
            fused = GAP(grads) * ( GAP(cosine_map) ** alignment_power )
        """
        # 1) 梯度 GAP 权重（Grad-CAM 经典做法）
        # activations: [B, K, H, W] -> numpy, grads 同形状
        grads_pooled = np.mean(grads, axis=(2, 3))  # [B, K]

        # 2) 余弦对齐权重（B-Cos）
        with torch.no_grad():
            # 把当前层的激活转成 torch（用于取得尺寸等；计算在 GPU 可选）
            act_t = torch.from_numpy(activations)
            if isinstance(input_tensor, torch.Tensor):
                act_t = act_t.to(input_tensor.device)

            cosine_map = self._compute_cosine_similarity_map(
                input_tensor, target_layer, act_t
            )  # [B, K, H, W]
            align_weights = torch.mean(cosine_map, dim=(2, 3))  # [B, K]
            align_weights = align_weights.clamp(min=-1.0, max=1.0)  # 余弦范围稳定
            align_weights = align_weights.cpu().numpy()

        # 3) 融合（可用指数加强/减弱对齐影响）
        fused_weights = grads_pooled * (np.power(align_weights, self.alignement_power))

        return fused_weights

    # def get_cam_image(self,
    #                   input_tensor: torch.Tensor,
    #                   target_layer: torch.nn.Module,
    #                   targets: List[torch.nn.Module],
    #                   activations: np.ndarray,
    #                   grads: np.ndarray,
    #                   eigen_smooth: bool = False) -> np.ndarray:
    #     weights = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads)
    #     weighted_activations = weights[:, :, None, None] * activations
    #
    #     if eigen_smooth:
    #         cam = get_2d_projection(weighted_activations)
    #     else:
    #         cam = weighted_activations.sum(axis=1)
    #     return cam

    def _compute_fused(self,
                       input_tensor: torch.Tensor,
                       target_layer: torch.nn.Module,
                       activations_np: np.ndarray,
                       grads_np: np.ndarray):
        """
        类条件融合：
          1) class-agnostic cosine_map -> class-conditional cosine_map_class
                cosine_map_class = cosine_map * sign(grads)   # 位置级
          2) A_mag = GAP(|cosine_map_class|)^power
             s     = sign(GAP(cosine_map_class))
          3) G_mag = GAP(|grads|)
          4) gate  = relu(s * cosine_map_class)  (逐通道归一化到 [0,1])
        """

        # 多标签融合模块
        min_vals = np.min(grads_np, axis=0)
        max_vals = np.max(grads_np, axis=0)
        if not np.array_equal(min_vals, max_vals):
            grads_ = 2 * (grads_np - max_vals) / (max_vals - min_vals + 1e-7) + 1
        else:
            grads_ = 0
        grads_nps = grads_ * grads_np + grads_ + grads_np

        # return np.mean(grads, axis=(2, 3))

        device = input_tensor.device
        B, K, H, W = activations_np.shape

        # (a) 梯度强度（幅度源之一）
        grads = torch.from_numpy(grads_nps).to(device)    # [B,K,H,W]
        G_mag = grads.abs().mean(dim=(2, 3))             # [B,K] >= 0

        # (b) 类无关余弦 → 类条件余弦
        act_t = torch.from_numpy(activations_np).to(device)
        cosine_map = self._compute_cosine_similarity_map(input_tensor, target_layer, act_t)  # [B,K,H,W]

        grad_sign = grads.sign()                         # 位置级符号
        cosine_map_class = cosine_map * grad_sign        # 把“反向传播方向”注入到对齐图

        # (c) 对齐强度 + 方向（通道级）
        A_mag = cosine_map_class.abs().mean(dim=(2, 3)).clamp(min=1e-6)   # [B,K]
        A_mag = A_mag.pow(self.alignement_power)
        s = cosine_map_class.mean(dim=(2, 3)).sign()                      # [B,K] ∈ {-1,0,1}

        # (d) 通道幅度（非负）
        amp = (G_mag * A_mag)                                             # [B,K] >= 0

        # (e) 空间门控（与通道整体方向一致的位置才放大）
        s_expand = s[:, :, None, None]
        gate = (s_expand * cosine_map_class).clamp(min=0.0)               # [B,K,H,W] >= 0

        # 使用稳健分位数归一化
        gate = self._robust_quantile_normalize(gate)

        # # 逐通道归一化 gate 到 [0,1]，避免数值不稳
        # gate_max = gate.flatten(2).amax(dim=2, keepdim=True).clamp(min=1e-6)  # [B,K,1]
        # gate = gate / gate_max.view(B, K, 1, 1)

        return amp, s, gate

    def _robust_quantile_normalize(self, gate, lower_quantile=0.1, upper_quantile=0.95):
        """稳健的分位数归一化"""
        B, K, H, W = gate.shape
        gate_flat = gate.flatten(2)

        with torch.no_grad():
            q_low = torch.quantile(gate_flat, lower_quantile, dim=2, keepdim=True)
            q_high = torch.quantile(gate_flat, upper_quantile, dim=2, keepdim=True).clamp(min=1e-6)

        # 基于分位数范围归一化
        gate_norm = (gate_flat - q_low) / (q_high - q_low)

        # 温和的截断，保留一些动态范围但避免极端值
        gate_norm = gate_norm.clamp(0, 2.0) / 2.0

        return gate_norm.view(B, K, H, W)

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: np.ndarray,
                      grads: np.ndarray,
                      eigen_smooth: bool = False) -> np.ndarray:
        amp, s, gate = self._compute_fused(input_tensor, target_layer, activations, grads)

        device = input_tensor.device
        A = torch.from_numpy(activations).to(device)     # [B,K,H,W]
        A_eff = A * gate                                 # 空间门控后的激活

        if self.signed:
            w = amp * s                                  # 可正可负（保留抑制证据）
        else:
            w = amp                                      # 只用强度

        cam = (w[:, :, None, None] * A_eff).sum(dim=1)   # [B,H,W]
        # cam = (w[:, :, None, None] * A).sum(dim=1)   # [B,H,W]

        if eigen_smooth:
            cam_eig = get_2d_projection((w[:, :, None, None] * A_eff).detach().cpu().numpy())
            cam = torch.from_numpy(cam_eig).to(device)

        if self.relu_at_end:
            cam = cam.clamp(min=0)

        return cam.detach().cpu().numpy()

    # 测试多层融合方法
    # 可选：自定义多层聚合（对齐后更偏重深层）
    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        result = cam_per_target_layer[-1][:, 0, ...]
        for cam in cam_per_target_layer[::-1][1:]:
            result = result * cam[:, 0, ...] + result
        return scale_cam_image(result)
