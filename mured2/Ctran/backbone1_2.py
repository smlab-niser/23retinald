import torch.nn as nn
import torchvision.models as models

class DenseNet201(nn.Module):
    def __init__(self, num_classes, embed_dim):
        self.embed_dim = embed_dim
        super(DenseNet201, self).__init__()

        self.features = models.densenet201(pretrained=True).features #Feature extractor all except the final layer
        self.num_features = embed_dim * 2
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(self.num_features, embed_dim) # new projection layer
        self.out_layer = nn.Linear(embed_dim, num_classes * embed_dim)

    def forward(self, x):
        print('Before featuring', x.shape)
        x = self.features(x)
        print('After featuring', x.shape)
        x = self.avgpool(x)
        print('After avgpooling', x.shape)
        x = x.view(x.size(0), -1)
        print('After viewing', x.shape)
        x = self.proj(x) # project the features to the desired embed_dim
        print('After projecting', x.shape)
        x = self.out_layer(x) # map to (batch_size, num_classes, embed_dim)
        print('After outlaying', x.shape)
        x = x.view(x.size(0), -1, self.embed_dim) # reshape to (batch_size, num_classes, embed_dim)
        print('After viewing againr:', x.shape)
        x = x.permute(1, 0, 2) # add the seq_length dimension
        print('After permuting', x.shape)
        return x
    
    
class DenseNet201b(nn.Module):
    def __init__(self, embed_dim):
        super(DenseNet201b, self).__init__()

        self.features = models.densenet201(pretrained=True).features # Train all previous layers, features implies the feature tensor
        self.num_features = self.features[-1].num_features # num_features stores number of features in the last output
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))   # Reduces spacial dimensions
        self.proj = nn.Linear(self.num_features, embed_dim) # Maps to create embedding of dimension embed_dim

    def forward(self, x):                 # x.shape = [batch, channel, height, width]
        x = self.features(x)              # x.shape = [batch, num_features, reduced_height, reduced_width]
        x = self.maxpool(x)               # x.shape = [batch, num_features, 1, 1]
        x = x.view(x.size(0), -1)         # x.shape = [batch, num_features], "-1" indicates the second dim is to be left intact
        x = self.proj(x)                  # x.shape = [batch, embed_dim]
        return x
    


