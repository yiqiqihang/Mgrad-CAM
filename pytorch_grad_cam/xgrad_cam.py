import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.image import scale_cam_image


class XGradCAM(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            use_cuda=False,
            reshape_transform=None):
        super(
            XGradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 1e-7
        weights = grads * activations / \
            (sum_activations[:, :, None, None] + eps)
        weights = weights.sum(axis=(2, 3))
        return weights

    # 测试多层融合方法
    def aggregate_multi_layers(
        self,
        cam_per_target_layer: np.ndarray) -> np.ndarray:
        result = cam_per_target_layer[-1][:, 0, ...]
        for cam in cam_per_target_layer[::-1][1:]:
            result = result * cam[:, 0, ...] + result
        return scale_cam_image(result)
