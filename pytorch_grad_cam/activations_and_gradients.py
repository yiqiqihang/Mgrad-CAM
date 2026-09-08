class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        # 初始化存储列表（目标层的梯度 and 激活值）
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        #初始化一个空列表，用于存储注册的钩子
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))
    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            # 这里将梯度放在列表开头，是因为在反向传播时，梯度是从最后层往最先层计算的，所以我们需要逆序存储。这样，当我们从头到尾遍历梯度列表时，顺序就和层的前向传播顺序一致。
            self.gradients = [grad.cpu().detach()] + self.gradients
        output.register_hook(_store_grad)
    #  这个方法的作用是：在每次调用时，清空之前存储的梯度和激活值，然后通过模型进行前向传播，并返回模型输出。
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        # 触发之前注册的前向钩子执行前向传播
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()
