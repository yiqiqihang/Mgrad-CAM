import tracemalloc
from typing import List

import cv2
import numpy as np
import torch
import gc

from pytorch_grad_cam import ActivationsAndGradients
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils import get_2d_projection
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# from pympler import tracker, summary, muppy


class Grad_CAM_NEW(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            Grad_CAM_NEW,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        # memory_tracker = tracker.SummaryTracker()
        # tracemalloc.start()
        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        target_size = self.get_target_width_height(input_tensor)
        outputs = self.model(input_tensor)
        logits = torch.sigmoid(outputs)
        assert len(targets) == input_tensor.shape[0]
        target_all_index = []
        target_all_list = []
        target_all_list_ = []
        for item in range(len(targets)):
            target_sigmoid = []
            for category in range(outputs.shape[-1]):
                if logits[item, category] > 0.5:
                    target_sigmoid.append(ClassifierOutputTarget(category))
            if targets is None:
                target_all = target_sigmoid
            else:
                x = [tar.category for tar in target_sigmoid]
                y = [targets[item].category]
                z = list(set(x).union(set(y)))
                target_all = [ClassifierOutputTarget(category) for category in z]
            target_all_index.append(list([tar.category for tar in target_all]).index(targets[item].category))
            target_all_list.append(target_all)
            target_all_list_.extend(target_all)

        assert len(target_all_list) == input_tensor.shape[0]

        input_list = []
        for item in range(input_tensor.shape[0]):
            input_list.extend([input_tensor[item] for _ in target_all_list[item]])
        input_tensor_all = torch.stack(input_list, dim=0)

        self.model.zero_grad()
        loss = sum([target(output)
                    for target, output in zip(target_all_list_, self.activations_and_grads(input_tensor_all))])
        loss.backward(retain_graph=True)

        self.activations_and_grads.gradients = self.get_grad_list(target_all_list, target_all_index)
        self.activations_and_grads.activations = self.get_acts_list(target_all_list, target_all_index)
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        aggregate_multi_layers = self.aggregate_multi_layers(cam_per_layer)
        aggregate_multi_layers = self.resize_tensor(aggregate_multi_layers, target_size[1],
                                                    target_size[0])[:, 0, ...].cpu().numpy()

        del loss, input_tensor_all
        gc.collect()
        # snapshot = tracemalloc.take_snapshot()
        # for stat in snapshot.statistics("lineno")[:3]:
        #     print(stat)
        return aggregate_multi_layers

    def get_grad_list(self, target_all_list, target_all_index):
        grad_list_all = []
        index_start = 0
        for item in range(len(target_all_list)):
            grads = [gr[index_start: index_start + len(target_all_list[item])].detach().data
                     for gr in self.activations_and_grads.gradients]
            grad_list = []
            for grad in grads:
                gradw = torch.clamp(grad, min=0.0)
                # grad_weights = self.mean_max_normalization_tensor(gradw)
                # grad_weights = torch.nn.functional.normalize(gradw, p=1, dim=0)
                grad_weights = torch.softmax(gradw, dim = 0)
                grad_ = grad_weights[target_all_index[item], ...] * grad[target_all_index[item], ...]
                # grad_ = grad[target_all_index[item], ...]
                # grad_ = self.min_max_normalization_tensor(grad_.unsqueeze(0))[0]
                grad_list.append(grad_)
            grad_list_all.append(grad_list)
            index_start += len(target_all_list[item])
        grad = [torch.stack([gr[j] for gr in grad_list_all], dim=0) for j in range(len(grad_list_all[0]))]
        return grad

    def get_acts_list(self, target_all_list, target_all_index):
        act_list_all = []
        index_start = 0
        for item in range(len(target_all_list)):
            acts = [act[index_start: index_start + len(target_all_list[item])].detach().data
                    for act in self.activations_and_grads.activations]
            act_list = [act[target_all_index[item], ...] for act in acts]
            act_list_all.append(act_list)
            index_start += len(target_all_list[item])
        act = [torch.stack([gr[j] for gr in act_list_all], dim=0) for j in range(len(act_list_all[0]))]
        return act

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        activations_list = [a.detach().data for a in self.activations_and_grads.activations]
        grads_list = [g.detach().data for g in self.activations_and_grads.gradients]

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
            cam = torch.clamp(cam, 0.0)
            scaled = self.min_max_normalization_tensor(cam)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = cam_per_target_layer[::-1]
        old = cam_per_target_layer[0]
        for item in range(1, len(cam_per_target_layer)):
            new = cam_per_target_layer[item]
            old_re = self.resize_tensor(old, new.shape[2], new.shape[3])
            old = old_re * new + old_re
        return self.min_max_normalization_tensor(old)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      targets,
                      activations,
                      grads,
                      eigen_smooth: bool = False) -> np.ndarray:

        weighted_activations = torch.mean(grads, dim=(2, 3), keepdim=True) * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    @staticmethod
    def resize_numpy(images, H, W):
        """
        param images: Numpy [B, C, H, W].
        param H: int
        param W: int
        """
        images = images.transpose([0, 2, 3, 1])
        image_resize_list = []
        for image in images:
            if images.shape[-1] == 1:
                image_resize = cv2.resize(image[..., 0], [W, H])
                image_resize_list.append(np.expand_dims(image_resize, axis=0))
            else:
                image_resize = cv2.resize(image, [W, H])
                image_resize_list.append(image_resize.transpose([2, 0, 1]))
        images_ = np.stack(image_resize_list, axis=0)
        return images_

    @staticmethod
    def resize_tensor(images, H, W):
        """
        param images: Numpy [B, C, H, W].
        param H: int
        param W: int
        """
        images_ = torch.nn.functional.interpolate(images, size=[H, W], align_corners=False, mode="bilinear")
        return images_

    # @staticmethod
    # def min_max_normalization_numpy(data):
    #     # data_list = []
    #     # for item in range(data.shape[0]):
    #     #     data_list.append((data - np.min(data[item])) / (np.max(data[item]) - np.min(data[item]) + 1e-7))
    #     # return np.concatenate(data_list, axis=0)
    #     data_ = (data - np.min(data, dim=0)) / (np.max(data, dim=0) - np.min(data, dim=0) + 1e-7)
    #     return data_

    @staticmethod
    def min_max_normalization_tensor(data):
        data_list = []
        for item in range(data.shape[0]):
            data_list.append(
                (data[item] - torch.min(data[item])) / (torch.max(data[item]) + 1e-7))
        if data.shape[0] != 1:
            return torch.stack(data_list, dim=0)
        else:
            return data_list[0].unsqueeze(0)
    
    @staticmethod
    def mean_max_normalization_tensor(data):
        data_list = []
        for item in range(data.shape[0]):
            data_list.append(
                (data[item] - torch.mean(data[item])) / (torch.max(data[item]) + 1e-7))
        if data.shape[0] != 1:
            return torch.stack(data_list, dim=0)
        else:
            return data_list[0].unsqueeze(0)
