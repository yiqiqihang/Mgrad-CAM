import torch.nn as nn
import torchvision.models as models

class VGG16_Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16_Backbone, self).__init__()

        vgg = models.vgg16(pretrained=pretrained)

        # 原始特征层
        self.features = vgg.features

        # 👇 在这里定义 block
        self.block3 = self.features[:16]   # block3
        self.block4 = self.features[16:23] # block4
        self.block5 = self.features[23:30] # block5

        self.classifier = vgg.classifier

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x