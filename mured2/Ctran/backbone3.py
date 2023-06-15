import torch.nn as nn
import torchvision.models as models

class EfficientNetV2Large(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super(EfficientNetV2Large, self).__init__()

        self.embed_dim = embed_dim
        self.num_features = embed_dim * 2

        self.features = models.efficientnet_v2_l(pretrained=True)
        self.features.conv_stem = nn.Conv2d(3, 80, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.features.bn1 = nn.BatchNorm2d(80, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(self.num_features, embed_dim)
        self.out_layer = nn.Linear(embed_dim, num_classes * embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        x = self.out_layer(x)
        x = x.view(x.size(0), -1, self.embed_dim)
        x = x.permute(1, 0, 2)
        return x