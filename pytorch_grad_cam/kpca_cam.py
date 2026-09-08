import tqdm
import torch
import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection_kernel
from pytorch_grad_cam.utils.image import scale_cam_image

class KPCACAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None,
                 kernel='sigmoid', gamma=0.1):
        super(KPCACAM, self).__init__(model,
                                       target_layers,
                                       use_cuda)
        self.kernel=kernel
        self.gamma=gamma

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations,
                        grads):
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(
                size=input_tensor.shape[-2:])
            activation_tensor = torch.from_numpy(activations)
            if self.cuda:
                activation_tensor = activation_tensor.cuda()

            upsampled = upsample(activation_tensor)

            maxs = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).min(dim=-1)[0]

            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins + 1e-7)

            input_tensors = input_tensor[:, None,
                            :, :] * upsampled[:, :, None, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            scores = []
            for target, tensor in zip(targets, input_tensors):
                for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE)):
                    batch = tensor[i: i + BATCH_SIZE, :]
                    outputs = [target(o).cpu().item()
                               for o in self.model(batch)]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(activations.shape[0], activations.shape[1])
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       target_category,
                                       activations,
                                       grads)
        activations = weights[:, :, None, None] * activations
        return get_2d_projection_kernel(activations, self.kernel, self.gamma)

    # 测试多层融合方法
    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        result = cam_per_target_layer[-1][:, 0, ...]
        for cam in cam_per_target_layer[::-1][1:]:
            result = result * cam[:, 0, ...] + result
        return scale_cam_image(result)

    # def get_cam_image(self,
    #                   input_tensor,
    #                   target_layer,
    #                   target_category,
    #                   activations,
    #                   grads,
    #                   eigen_smooth):
    #     return get_2d_projection_kernel(grads * activations, self.kernel, self.gamma)
