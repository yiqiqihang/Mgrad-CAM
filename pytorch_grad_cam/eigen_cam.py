import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.image import scale_cam_image

# https://arxiv.org/abs/2008.00299


class EigenCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(EigenCAM, self).__init__(model,
                                       target_layers,
                                       use_cuda,
                                       reshape_transform,
                                       uses_gradients=False)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)

    # 测试多层融合方法
    def aggregate_multi_layers(
        self,
        cam_per_target_layer: np.ndarray) -> np.ndarray:
        result = cam_per_target_layer[-1][:, 0, ...]
        for cam in cam_per_target_layer[::-1][1:]:
            result = result * cam[:, 0, ...] + result
        return scale_cam_image(result)
