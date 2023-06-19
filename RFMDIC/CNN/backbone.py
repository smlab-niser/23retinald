import torchvision.models as models
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import timm

class DenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()
        
        self.features = models.densenet121(pretrained=True).features
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DenseNet201(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet201, self).__init__()
        
        self.features = models.densenet201(pretrained=True).features
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1920, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class ResNet152V2(nn.Module):
    def __init__(self, num_classes):
        super(ResNet152V2, self).__init__()

        self.model = timm.create_model("resnet152d", pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class EfficientNetV2Small(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetV2Small, self).__init__()
        
        self.model = timm.create_model("tf_efficientnetv2_s", pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

# Dictionary containing the models
backbone = {
    'densenet121': DenseNet121,
    'densenet201': DenseNet201,
    'resnet152v2': ResNet152V2,
    'efficientnetv2_extralarge': EfficientNetV2Small,
}