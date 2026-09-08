from typing import List
import numpy as np
import matplotlib as plt

from pytorch_grad_cam import ActivationsAndGradients
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection_kernel


# from pympler import tracker, summary, muppy


class MGradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None, kernel='sigmoid', gamma=0.1):
        super(
            MGradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

        self.kernel = kernel
        self.gamma = gamma

    # max-min
    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        min_vals = np.min(grads, axis=0)
        max_vals = np.max(grads, axis=0)
        if not np.array_equal(min_vals, max_vals):
            grads_nore = (grads - min_vals) / (max_vals - min_vals + 1e-7)
        else:
            grads_nore = 0
        grads = grads * (1 + grads_nore)

        return np.mean(grads, axis=(2, 3))
        # return grads


    def aggregate_multi_layers(
            self,
            cam_per_target_layer: np.ndarray) -> np.ndarray:
        result = cam_per_target_layer[-1][:, 0, ...]
        for cam in cam_per_target_layer[::-1][1:]:
            # result = result * cam[:, 0, ...] + result
            result = result * cam[:, 0, ...] + result
        return scale_cam_image(result)

    # softmax across class dimension
    # def get_cam_weights(self,
    #                     input_tensor,
    #                     target_layer,
    #                     target_category,
    #                     activations,
    #                     grads):
    #     # Softmax normalization across class/batch dimension
    #     # 为了避免数值溢出，减去最大值
    #     max_vals = np.max(grads, axis=0)
    #     exp_grads = np.exp(grads - max_vals)
    #     sum_exp_grads = np.sum(exp_grads, axis=0)
    #
    #     grads_nore = exp_grads / (sum_exp_grads + 1e-7)
    #
    #     grads = grads * (1 + grads_nore)
    #
    #     return np.mean(grads, axis=(2, 3))

    # Z-score
    # def get_cam_weights(self,
    #                     input_tensor,
    #                     target_layer,
    #                     target_category,
    #                     activations,
    #                     grads):
    #     # Z-score normalization (Standardization)
    #     mean_vals = np.mean(grads, axis=0)
    #     std_vals = np.std(grads, axis=0)
    #
    #     # 避免除以0
    #     grads_nore = (grads - mean_vals) / (std_vals + 1e-7)
    #
    #     grads = grads * (1 + grads_nore)
    #
    #     return np.mean(grads, axis=(2, 3))

    # 测试多层融合方法