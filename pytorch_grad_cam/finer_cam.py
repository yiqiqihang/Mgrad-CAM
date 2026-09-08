import numpy as np
import torch
from typing import List
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.model_targets import FinerWeightedTarget
from pytorch_grad_cam.utils.image import scale_cam_image


# Finer-CAM: https://arxiv.org/pdf/2501.11309


class FinerCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None,detach = True):
        super(FinerCAM, self).__init__(model,
                                       target_layers,
                                       use_cuda,
                                       reshape_transform)

        self.detach = detach

        # self.base_cam = base_method(model, target_layers, use_cuda, reshape_transform)
        # self.compute_input_gradient = self.base_cam.compute_input_gradient
        # self.uses_gradients = self.base_cam.uses_gradients

    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False) -> np.ndarray:
        return self.forward(input_tensor, targets, eigen_smooth)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))

    # 测试多层融合方法
    def aggregate_multi_layers(
            self,
            cam_per_target_layer: np.ndarray) -> np.ndarray:
        result = cam_per_target_layer[-1][:, 0, ...]
        for cam in cam_per_target_layer[::-1][1:]:
            # result = result * cam[:, 0, ...] + result
            result = result * cam[:, 0, ...] + result
        return scale_cam_image(result)

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module] = None,
                target_size=None,
                eigen_smooth: bool = False,
                alpha: float = 1,
                comparison_categories: List[int] = [1, 2, 3],
                target_idx: int = None,
                H: int = None,
                W: int = None
                ) -> np.ndarray:

        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)

        # 关键修改：将 ClassifierOutputTarget 转换为 FinerWeightedTarget
        if targets is not None and all(hasattr(t, 'category') for t in targets):
            # 如果传入的是 ClassifierOutputTarget，转换为 FinerWeightedTarget
            output_data = outputs.detach().cpu().numpy()

            # 获取所有目标类别
            main_categories = [t.category for t in targets]

            # 为每个目标类别创建 FinerWeightedTarget
            finer_targets = []
            for i, main_category in enumerate(main_categories):
                # 选择对比类别：排除当前主类别，选择其他目标类别作为对比
                comparison_cats = [cat for cat in main_categories if cat != main_category]

                # 如果对比类别不足，从预测结果中补充
                if len(comparison_cats) < len(comparison_categories):
                    # 获取当前样本的预测概率
                    probs = output_data[i]
                    # 选择与主类别不同的高概率类别作为补充
                    other_indices = np.argsort(probs)[::-1]
                    other_indices = [idx for idx in other_indices if
                                     idx != main_category and idx not in comparison_cats]
                    # 补充到需要的数量
                    needed = len(comparison_categories) - len(comparison_cats)
                    comparison_cats.extend(other_indices[:needed])

                # 确保对比类别数量一致
                comparison_cats = comparison_cats[:len(comparison_categories)]

                target = FinerWeightedTarget(main_category, comparison_cats, alpha)
                finer_targets.append(target)

            targets = finer_targets
            # print(
            #     f"转换后的 FinerWeightedTargets: 主类别 {[t.main_category for t in targets]}, 对比类别 {[t.comparison_categories for t in targets]}")

        elif targets is None:
            # 自动创建 FinerWeightedTarget 的逻辑
            output_data = outputs.detach().cpu().numpy()
            target_logits = np.max(output_data, axis=-1) if target_idx is None else output_data[:, target_idx]
            sorted_indices = np.argsort(np.abs(output_data - target_logits[:, None]), axis=-1)
            targets = [FinerWeightedTarget(int(sorted_indices[i, 0]),
                                           [int(sorted_indices[i, idx]) for idx in comparison_categories],
                                           alpha)
                       for i in range(output_data.shape[0])]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            if self.detach:
                loss.backward(retain_graph=True)
            else:
                # keep the computational graph, create_graph = True is needed for hvp
                torch.autograd.grad(loss, input_tensor, retain_graph=True, create_graph=True)
                # When using the following loss.backward() method, a warning is raised: "UserWarning: Using backward() with create_graph=True will create a reference cycle"
                # loss.backward(retain_graph=True, create_graph=True)
            if 'hpu' in str(self.device):
                self.__htcore.mark_step()

        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)