import numpy as np
import torch
import tqdm
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.image import scale_cam_image


class ScoreCAMNew(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            use_cuda=False,
            reshape_transform=None):
        super(ScoreCAMNew, self).__init__(model,
                                       target_layers,
                                       use_cuda,
                                       reshape_transform=reshape_transform,
                                       uses_gradients=False)

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
            scores_ = []
            for target, tensor in zip(targets, input_tensors):
                for i in range(0, tensor.size(0), BATCH_SIZE):
                    batch = tensor[i: i + BATCH_SIZE, :]
                    output_raw = [o for o in self.model(batch)]
                    outputs = [target(o).cpu().item() for o in output_raw]
                    outputs_ = [self.get_scores(o, target).cpu().item() for o in output_raw]
                    scores.extend(outputs)
                    scores_.extend(outputs_)
            scores = torch.Tensor(scores)
            scores_ = torch.Tensor(scores_)
            scores = scores.view(activations.shape[0], activations.shape[1])
            scores_ = scores_.view(activations.shape[0], activations.shape[1])
            # weights = torch.nn.functional.normalize(scores * scores_, p=1, dim=-1).numpy()
            weights = torch.nn.Softmax(dim=-1)(scores * scores_).numpy()
            return weights

    @staticmethod
    def get_scores(output, target):
        output = torch.sigmoid(output)
        indices = list(set(torch.where(output > 0.5)[0].tolist()).union(set([target.category])))
        index = indices.index(target.category)
        scores = torch.nn.Softmax(dim=-1)(output[indices])[index]
        # scores = torch.nn.functional.normalize(output[indices], p=1, dim=-1)[index]
        return scores

    def aggregate_multi_layers(
            self,
            cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = cam_per_target_layer[::-1]
        cam_per_target_layer = [torch.from_numpy(o) for o in cam_per_target_layer]
        old = cam_per_target_layer[0]
        for item in range(1, len(cam_per_target_layer)):
            new = cam_per_target_layer[item]
            old_re = self.resize_tensor(old, new.shape[2], new.shape[3])
            old = old_re * new + old_re
        return self.min_max_normalization_tensor(old).cpu().numpy()[:, 0, ...]

    @staticmethod
    def min_max_normalization_tensor(data):
        data_list = []
        for item in range(data.shape[0]):
            data_list.append(
                (data[item] - torch.min(data[item])) / (torch.max(data[item]) - torch.min(data[item]) + 1e-7))
        if data.shape[0] != 1:
            return torch.stack(data_list, dim=0)
        else:
            return data_list[0].unsqueeze(0)

    @staticmethod
    def resize_tensor(images, H, W):
        """
        param images: Numpy [B, C, H, W].
        param H: int
        param W: int
        """
        images_ = torch.nn.functional.interpolate(images, size=[H, W], align_corners=False, mode="bilinear")
        return images_


