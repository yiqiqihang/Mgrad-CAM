import numpy as np
import torch
import torch.nn.functional as F
import ttach as tta
from typing import Callable, List, Tuple, Dict, Optional
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class BaseCAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True,
                 alignement_power: float = 1.0) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        # 设置ActivationsAndGradients类的实例。初始化该函数，即传参进入该函数ActivationsAndGradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

        # # ------------ NEW: 为“对齐权重”做准备 ------------
        # # 把 target_layer 解析到一个“真正进行卷积”的 nn.Conv2d（若 target 本身是 Conv2d 就直接用它，
        # # 若是 Sequential/Bottleneck，则取内部“最后一个卷积层”）。
        # # 控制对齐权重的强度
        # self.alignement_power = alignement_power
        # self._target_to_conv: Dict[torch.nn.Module, torch.nn.Conv2d] = {}
        # self._layer_inputs: Dict[torch.nn.Module, torch.Tensor] = {}
        # self._input_hooks: List[torch.utils.hooks.RemovableHandle] = []
        #
        # for tl in self.target_layers:
        #     conv = self._resolve_conv(tl)
        #     self._target_to_conv[tl] = conv
        #
        #     # 捕获该 conv 的前向输入（即 X，用于后续 unfold 成 patch）
        #     def _save_input(module, inp):
        #         # inp 是一个 tuple，取第一个张量 [B, Cin, Hin, Win]
        #         self._layer_inputs[module] = inp[0]
        #
        #     h = conv.register_forward_pre_hook(_save_input)
        #     self._input_hooks.append(h)


    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    # 下述的代码是关于alignGradCAM的，如果不注释掉会影响FullGrad的使用
    # # ------------ NEW: 解析 target_layer 到一个 Conv2d ------------
    # def _resolve_conv(self, layer: torch.nn.Module) -> torch.nn.Conv2d:
    #     if isinstance(layer, torch.nn.Conv2d):
    #         return layer
    #     # 找到“最后一个” Conv2d（更贴近输出）
    #     last_conv = None
    #     for m in layer.modules():
    #         if isinstance(m, torch.nn.Conv2d):
    #             last_conv = m
    #     if last_conv is None:
    #         raise RuntimeError(
    #             "Align-Grad-CAM 需要在目标层（或其子模块）里找到 nn.Conv2d，"
    #             "请把 target_layers 设为卷积层或包含卷积的模块（如 ResNet 的 layer4）。"
    #         )
    #     return last_conv
    #
    # # ------------ NEW: 计算局部余弦相似度图（B-Cos） ------------
    # @torch.no_grad()
    # def _compute_cosine_similarity_map(self,
    #                                    input_tensor: torch.Tensor,
    #                                    target_layer: torch.nn.Module,
    #                                    activations_t: torch.Tensor
    #                                    ) -> torch.Tensor:
    #     """
    #     返回形状 [B, K, Hout, Wout] 的 cosine map，
    #     K 为“对齐 Conv”的输出通道数（通常与 activations 的通道一致或接近）。
    #     """
    #     # 选定要对齐的真实卷积层
    #     conv = self._target_to_conv[target_layer]
    #     if conv not in self._layer_inputs:
    #         raise RuntimeError("未捕获到对齐卷积的输入，请确认 forward 已执行并且 hook 正常生效。")
    #
    #     x = self._layer_inputs[conv]            # [B, Cin, Hin, Win]
    #     w = conv.weight                         # [K, Cin, kH, kW]
    #     # B = x.shape[0]
    #
    #     # 卷积超参
    #     kH, kW = conv.kernel_size
    #     sH, sW = conv.stride
    #     pH, pW = conv.padding
    #     dH, dW = conv.dilation
    #
    #     # 用与该卷积相同的参数把输入展开成局部补丁：[B, Cin*kH*kW, L]，L = Hout*Wout
    #     patches = F.unfold(x, kernel_size=(kH, kW),
    #                        dilation=(dH, dW),
    #                        padding=(pH, pW),
    #                        stride=(sH, sW))                         # [B, Cin*kH*kW, L]
    #     # L = patches.shape[-1]
    #
    #     # 卷积核拉平成向量：[K, Cin*kH*kW]
    #     w_flat = w.view(w.shape[0], -1)                             # [K, Cin*kH*kW]
    #
    #     # 单位化（避免数值不稳）
    #     eps = 1e-6
    #     w_norm = w_flat / (w_flat.norm(dim=1, keepdim=True) + eps)  # [K, Cin*kH*kW]
    #     p_norm = patches / (patches.norm(dim=1, keepdim=True) + eps)# [B, Cin*kH*kW, L]
    #
    #     # 逐位置余弦相似度： [B, K, L]
    #     cos_BKL = torch.einsum('bkL,Kk->bKL', p_norm, w_norm)
    #
    #     # reshape 到 [B, K, Hout, Wout]
    #     # Hout, Wout 可从 activations_t 直接读（更稳）
    #     Hout, Wout = activations_t.shape[-2], activations_t.shape[-1]
    #     cosine_map = cos_BKL.view(x.size(0), w.shape[0], Hout, Wout)
    #
    #     return cosine_map

    # def _compute_cosine_similarity_map(self,
    #                                    input_tensor: torch.Tensor,
    #                                    # target_layer: torch.nn.Module,
    #                                    target_layer: List[torch.nn.Module],
    #                                    activations: torch.Tensor) -> torch.Tensor:
    #     """
    #             计算余弦相似度图 (Cosine Similarity Map)
    #
    #             Args:
    #                 input_tensor: 输入张量 [B, C, H, W]
    #                 target_layer: 目标层
    #                 activations: 目标层的激活值 [B, K, H, W]
    #
    #             Returns:
    #                 cosine_map: 余弦相似度图 [B, K, H, W]
    #             """
    #     batch_size, num_channels, height, width = input_tensor.shape
    #     _, num_filters, act_height, act_width = activations.shape
    #
    #     cosine_map = torch.zeros_like(activations)
    #
    #     # 获取目标层的权重
    #     if hasattr(target_layer, "weight"):
    #         # [K, C ,kH, kW]
    #         weights = target_layer.weight.data
    #     else:
    #         # 对于没有weight属性的层，返回全1的映射（退化为原始Grad-CAM）
    #         print(f"Warning: Target layer {type(target_layer)} has no weight attribute. Using standard Grad-CAM.")
    #         return torch.ones_like(activations)
    #
    #      # 获取卷积核大小
    #     if hasattr(target_layer, "kernel_size"):
    #         kernel_size = target_layer.kernel_size
    #         if isinstance(kernel_size, tuple):
    #             kH, kW = kernel_size
    #         else:
    #             kH= kW = kernel_size
    #     else:
    #         kH = kW = 3
    #
    #     # 计算步长和填充
    #     stride = target_layer.stride if hasattr(target_layer, "stride") else (1,1)
    #     padding = target_layer.padding if hasattr(target_layer, "padding") else (0,0)
    #
    #     # 对每个批次、每个滤波器、每个空间位置计算余弦相似度
    #     for b in range(batch_size):
    #         for k in range(num_filters):
    #             for i in range(act_height):
    #                 for j in range(act_width):
    #                     # 计算输入图像中对应的patch位置
    #                     h_start = i * stride[0] - padding[0]
    #                     w_start = j * stride[1] - padding[1]
    #                     h_end = h_start + kH
    #                     w_end = w_start + kW
    #
    #                     # 处理边界情况
    #                     if h_start < 0 or w_start < 0 or h_end > height or w_end > width:
    #                         # 对于边界位置，使用零填充
    #                         patch = torch.zeros((num_channels,kH,kW),
    #                                             device=input_tensor.device,)
    #                         valid_h_start = max(h_start, 0)
    #                         valid_w_start = max(w_start, 0)
    #                         valid_h_end = min(h_end, height)
    #                         valid_w_end = min(w_end, width)
    #
    #                         patch_h_start = valid_h_start - h_start
    #                         patch_w_start = valid_w_start - w_start
    #                         patch_h_end = valid_h_start +(valid_h_end - valid_h_start)
    #                         patch_w_end = valid_h_end + (valid_w_end - valid_w_start)
    #
    #                         patch[:, patch_h_start:patch_h_end, patch_w_start:patch_w_end] = \
    #                             input_tensor[b, :, valid_h_start:valid_h_end, valid_w_start:valid_w_end]
    #                     else:
    #                         patch = input_tensor[b, :, h_start:h_end, w_start:w_end]
    #
    #                     # 获得当前滤波器的权重 [C ,kH, kW]
    #                     weight = weights[k]
    #
    #                     # 展平patch和权重以计算余弦相似度
    #                     patch_flat = patch.reshape(-1)
    #                     weight_flat = weight.reshape(-1)
    #
    #                     # 计算余弦相似度
    #                     cosine_sim = F.cosine_similarity(patch_flat.unsqueeze[0], weight_flat.unsqueeze[0],dim=1)
    #                     cosine_map[b, k, i, j] = cosine_sim
    #
    #     return cosine_map

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        targets: List[torch.nn.Module],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:
        # 如果采用了GPU则将input_tensor用GPU处理
        if self.cuda:
            input_tensor = input_tensor.cuda()
        # 判断是否需要计算输入图像的梯度
        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)
        # 调用该实例，其作用是调用之前注册的钩子获取梯度/激活信息(前向：会触发 hooks，捕获 activations、gradients，以及对齐卷积的 inputs)
        outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            target_categories = np.argmax(outputs.cpu().numpy(), axis=-1)
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]
        # 判断-->True-->①将模型梯度清零，防止梯度积累；②计算损失；③反向传播计算梯度，retain_graph=True表示保留计算图
        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output)
                        for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.

        # 每层显著性图
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        # 多层聚合（默认平均）
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    # 计算每一层的显著性图
    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(
            self,
            cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    # TTA平滑
    def forward_augmentation_smoothing(self,
                                       input_tensor: torch.Tensor,
                                       targets: List[torch.nn.Module],
                                       eigen_smooth: bool = False) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               targets,
                               eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    # 当我们在调用实例的时候就是先调用这个def __call__(self)方法
    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False) -> np.ndarray:

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor, targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()
        # 移除注册的hook,避免内存泄露
        try:
            for h in self._input_hooks:
                h.remove()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
