# import numpy as np
# import torch
# from pytorch_grad_cam.base_cam import BaseCAM
# from pytorch_grad_cam.utils.find_layers import find_layer_predicate_recursive
# from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
# from pytorch_grad_cam.utils.image import scale_accross_batch_and_channels, scale_cam_image
#
#
# # https://arxiv.org/abs/1905.00780
#
#
# class FullGrad(BaseCAM):
#     def __init__(self, model, target_layers=None, use_cuda=False, reshape_transform=None):
#         # ⚠ 传空列表给父类，跳过 BaseCAM 的卷积检查
#         BaseCAM.__init__(
#             self,
#             model=model,
#             target_layers=[],
#             use_cuda=use_cuda,
#             reshape_transform=reshape_transform,
#             compute_input_gradient=True
#         )
#
#         # 查找所有具有 bias 的 Conv2d / BatchNorm2d 层
#         def layer_with_2D_bias(layer):
#             if isinstance(layer, (torch.nn.Conv2d, torch.nn.BatchNorm2d)) and getattr(layer, 'bias', None) is not None:
#                 return True
#             return False
#
#         self.bias_layers = find_layer_predicate_recursive(model, layer_with_2D_bias)
#
#         if len(self.bias_layers) == 0:
#             raise RuntimeError("FullGrad: 未找到任何具有 bias 的层，请检查模型。")
#
#         # 保存 bias 数据
#         self.bias_data = [self.get_bias_data(layer).cpu().numpy() for layer in self.bias_layers]
#
#     def get_bias_data(self, layer):
#         """获取层的 bias 数据，用于 saliency 计算"""
#         if isinstance(layer, torch.nn.BatchNorm2d):
#             bias = - (layer.running_mean * layer.weight / torch.sqrt(layer.running_var + layer.eps)) + layer.bias
#             return bias.data
#         else:
#             return layer.bias.data
#
#     def compute_cam_per_layer(self, input_tensor, target_category, eigen_smooth):
#         """计算每一层的 CAM"""
#         input_grad = input_tensor.grad.data.cpu().numpy()
#         grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
#         cam_per_target_layer = []
#
#         target_size = self.get_target_width_height(input_tensor)
#
#         # 输入梯度乘输入
#         gradient_multiplied_input = np.abs(input_grad * input_tensor.data.cpu().numpy())
#         gradient_multiplied_input = scale_accross_batch_and_channels(gradient_multiplied_input, target_size)
#         cam_per_target_layer.append(gradient_multiplied_input)
#
#         # 遍历 bias 层
#         for bias, grads in zip(self.bias_data, grads_list):
#             bias = bias[None, :, None, None]
#             bias_grad = np.abs(bias * grads)
#             result = scale_accross_batch_and_channels(bias_grad, target_size)
#             result = np.sum(result, axis=1)
#             cam_per_target_layer.append(result[:, None, :])
#
#         cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
#
#         if eigen_smooth:
#             cam_per_target_layer = scale_accross_batch_and_channels(
#                 cam_per_target_layer, (target_size[0] // 8, target_size[1] // 8)
#             )
#             cam_per_target_layer = get_2d_projection(cam_per_target_layer)
#             cam_per_target_layer = cam_per_target_layer[:, None, :, :]
#             cam_per_target_layer = scale_accross_batch_and_channels(cam_per_target_layer, target_size)
#         else:
#             cam_per_target_layer = np.sum(cam_per_target_layer, axis=1)[:, None, :]
#
#         return cam_per_target_layer
#
#     def aggregate_multi_layers(self, cam_per_target_layer):
#         """将各层 CAM 聚合"""
#         result = np.sum(cam_per_target_layer, axis=1)
#         return scale_cam_image(result)


import numpy as np
import torch
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.find_layers import find_layer_predicate_recursive
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.image import scale_accross_batch_and_channels, scale_cam_image

# https://arxiv.org/abs/1905.00780


class FullGrad(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        if len(target_layers) > 0:
            print(
                "Warning: target_layers is ignored in FullGrad. All bias layers will be used instead")

        def layer_with_2D_bias(layer):
            bias_target_layers = [torch.nn.Conv2d, torch.nn.BatchNorm2d]
            if type(layer) in bias_target_layers and layer.bias is not None:
                return True
            return False
        target_layers = find_layer_predicate_recursive(
            model, layer_with_2D_bias)
        super(
            FullGrad,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform,
            compute_input_gradient=True)
        self.bias_data = [self.get_bias_data(
            layer).cpu().numpy() for layer in target_layers]

    def get_bias_data(self, layer):
        # Borrowed from official paper impl:
        # https://github.com/idiap/fullgrad-saliency/blob/master/saliency/tensor_extractor.py#L47
        if isinstance(layer, torch.nn.BatchNorm2d):
            bias = - (layer.running_mean * layer.weight
                      / torch.sqrt(layer.running_var + layer.eps)) + layer.bias
            return bias.data
        else:
            return layer.bias.data

    def compute_cam_per_layer(
            self,
            input_tensor,
            target_category,
            eigen_smooth):
        input_grad = input_tensor.grad.data.cpu().numpy()
        grads_list = [g.cpu().data.numpy() for g in
                      self.activations_and_grads.gradients]
        cam_per_target_layer = []
        target_size = self.get_target_width_height(input_tensor)

        gradient_multiplied_input = input_grad * input_tensor.data.cpu().numpy()
        gradient_multiplied_input = np.abs(gradient_multiplied_input)
        gradient_multiplied_input = scale_accross_batch_and_channels(
            gradient_multiplied_input,
            target_size)
        cam_per_target_layer.append(gradient_multiplied_input)

        # Loop over the saliency image from every layer
        assert(len(self.bias_data) == len(grads_list))
        for bias, grads in zip(self.bias_data, grads_list):
            bias = bias[None, :, None, None]
            # In the paper they take the absolute value,
            # but possibily taking only the positive gradients will work
            # better.
            bias_grad = np.abs(bias * grads)
            result = scale_accross_batch_and_channels(
                bias_grad, target_size)
            result = np.sum(result, axis=1)
            cam_per_target_layer.append(result[:, None, :])
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        if eigen_smooth:
            # Resize to a smaller image, since this method typically has a very large number of channels,
            # and then consumes a lot of memory
            cam_per_target_layer = scale_accross_batch_and_channels(
                cam_per_target_layer, (target_size[0] // 8, target_size[1] // 8))
            cam_per_target_layer = get_2d_projection(cam_per_target_layer)
            cam_per_target_layer = cam_per_target_layer[:, None, :, :]
            cam_per_target_layer = scale_accross_batch_and_channels(
                cam_per_target_layer,
                target_size)
        else:
            cam_per_target_layer = np.sum(
                cam_per_target_layer, axis=1)[:, None, :]

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        result = np.sum(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

