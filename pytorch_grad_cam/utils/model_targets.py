import numpy as np
import torch
import torchvision


class ClassifierOutputTarget:
    # 存储类别整数
    def __init__(self, category):
        self.category = category

    # 返回指定类别self.category的分数（分一维跟二维数组的输入形式）
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


class ClassifierOutputSoftmaxTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return torch.softmax(model_output, dim=-1)[self.category]
        return torch.softmax(model_output, dim=-1)[:, self.category]


class ClassifierOutputSigmoidTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return torch.sigmoid(model_output)[self.category]
        return torch.sigmoid(model_output)[:, self.category]

class BinaryClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if self.category == 1:
            sign = 1
        else:
            sign = -1
        return model_output * sign


class SoftmaxOutputTarget:
    def __init__(self):
        pass

    def __call__(self, model_output):
        return torch.softmax(model_output, dim=-1)


class RawScoresOutputTarget:
    def __init__(self):
        pass

    def __call__(self, model_output):
        return model_output


class SemanticSegmentationTarget:
    """ Gets a binary spatial mask and a category,
        And return the sum of the category scores,
        of the pixels in the mask. """

    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


class FasterRCNNBoxScoreTarget:
    """ For every original detected bounding box specified in "bounding boxes",
        assign a score on how the current bounding boxes match it,
            1. In IOU
            2. In the classification score.
        If there is not a large enough overlap, or the category changed,
        assign a score of 0.

        The total score is the sum of all the box scores.
    """

    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()

        if len(model_outputs["boxes"]) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            ious = torchvision.ops.box_iou(box, model_outputs["boxes"])
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and model_outputs["labels"][index] == label:
                score = ious[0, index] + model_outputs["scores"][index]
                output = output + score
        return output


class FinerWeightedTarget:
    """
    Computes a weighted difference between a primary category and a set of comparison categories.

    This target calculates the difference between the score for the main category and each of the comparison categories.
    It obtains a weight for each comparison category from the softmax probabilities of the model output and computes a
    weighted difference scaled by a comparison strength factor alpha.
    """

    def __init__(self, main_category, comparison_categories, alpha):
        self.main_category = main_category
        self.comparison_categories = comparison_categories
        self.alpha = alpha

    def __call__(self, model_output):
        select = lambda idx: model_output[idx] if model_output.ndim == 1 else model_output[..., idx]

        wn = select(self.main_category)

        prob = torch.softmax(model_output, dim=-1)

        weights = [prob[idx] if model_output.ndim == 1 else prob[..., idx] for idx in self.comparison_categories]
        numerator = sum(w * (wn - self.alpha * select(idx)) for w, idx in zip(weights, self.comparison_categories))
        denominator = sum(weights)

        return numerator / (denominator + 1e-9)