import torch.nn as nn
import torchvision.models as models
import timm
 
class ResNet152d(nn.Module):
    def __init__(self, num_classes, embed_dim):
        self.embed_dim = embed_dim
        super(ResNet152d, self).__init__()

        self.features = timm.create_model('resnet152d', pretrained=True)
        self.num_features = self.features.fc.in_features
        self.features.fc = nn.Identity() # remove the original fully connected layer
        self.proj = nn.Linear(self.num_features, embed_dim) # new projection layer
        self.out_layer = nn.Linear(embed_dim, num_classes * embed_dim)

    def forward(self, x):
        # print('Before featuring', x.shape)
        x = self.features(x)
        # print('After featuring', x.shape)
        x = x.view(x.size(0), -1)
        # print('After viewing', x.shape)
        x = self.proj(x) # project the features to the desired embed_dim
        # print('After projecting', x.shape)
        x = self.out_layer(x) # map to (batch_size, num_classes, embed_dim)
        # print('After outlaying', x.shape)
        x = x.view(x.size(0), -1, self.embed_dim) # reshape to (batch_size, num_classes, embed_dim)
        # print('After viewing againr:', x.shape)
        x = x.permute(1, 0, 2) # add the seq_length dimension
        # print('After permuting', x.shape)
        return x
    


