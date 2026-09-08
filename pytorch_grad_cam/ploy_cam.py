import torch
import torch.nn.functional as F
import math
from pytorch_grad_cam.utils.utils import *
from pytorch_grad_cam._poly_cam import _PolyCAM

class PCAMpm(_PolyCAM):
    """
    PCAM+/- variation of PolyCAM

    Init arguments:
        - model: model used for saliency map generation
        - target_layer_list: list of target layers for the iterative refinement process of PolyCAM
                    the list is to provide in the same order as forward pass
                    optional, automatically determined if not provided
                    example for vgg16, ['features.3', 'features.8', 'features.15', 'features.22', 'features.29']
        - batch_size: batch size used

    Call:
        - arguments:
            - input: input tensor to generate the saliency map
            - class_idx: index of the class to explain in the softmax output
        - return:
            list of saliency maps for each target layer
    """
    def __init__(self, model, target_layers=None, use_cuda = False, batch_size=32, intermediate_maps=False, lnorm=True):
        super(PCAMpm, self).__init__(model, target_layer_list=target_layers, intermediate_maps=intermediate_maps, lnorm=lnorm)
        self.batch_size = batch_size

    def get_weights(self, image_input, layer):
        with torch.no_grad():
            # first pass to store base score using hook
            _ = self.model(image_input)
            baseline_score = self.score

            # generate masks from activation map
            masks = self.normalize(self.target_activations[layer])
            masks = F.interpolate(masks, image_input.shape[-2:], mode='bilinear')
            masks = masks.transpose(0, 1)
            reverse_masks = 1 - masks

            # keep scores and reverse scores for weights computation
            scores = torch.zeros(masks.shape[0], *baseline_score.shape[-2:]).to(image_input.device)
            reverse_scores = torch.zeros(masks.shape[0], *baseline_score.shape[-2:]).to(image_input.device)

            # compute scores by batch
            for idx in range(math.ceil(scores.shape[0] / self.batch_size)):
                selection_slice = slice(idx * self.batch_size, min((idx + 1) * self.batch_size, scores.shape[0]))
                with torch.no_grad():
                    # forward pass with masked image, value stored using hook
                    self.model(image_input * masks[selection_slice])
                    score = self.score
                    # forward pass with reverse masked image, value stored using hook
                    self.model(image_input * reverse_masks[selection_slice])
                    reverse_score = self.score
                    # store two scores
                    scores[selection_slice] = score
                    reverse_scores[selection_slice] = reverse_score

            # compute weights from scores
            weights = scores + baseline_score - reverse_scores
            weights = torch.relu(weights)
            weights = weights.unsqueeze(0)
            return weights
