import torch.nn as nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class CTranEncoder(nn.Module):
    def __init__(self, num_classes, embed_dim, num_layers, num_heads, backbone):
        super(CTranEncoder, self).__init__()
        
        # Initialize the backbone network
        self.backbone = backbone
        
        # Initialize the TransformerEncoder
        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=2048)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
       
        # Initialize the final classification layer
        self.fc = nn.Linear(embed_dim, num_classes)
        
        # Initialize the projection layer
        self.proj = nn.Linear(2*embed_dim, embed_dim)
        
    def forward(self, x): 
        # Pass the input through the backbone network
        # print('Before backbone', x.shape)
        x = self.backbone(x)
        # print('After backbone', x.shape)
        
        # Pass the output through the TransformerEncoder
        x = self.transformer(x)
        # print('After transformer', x.shape)
        
        # Concatenate the first and last hidden states along the last dimension
        x = torch.cat((x[0], x[-1]), dim=-1)
        # print('After concatenation', x.shape)
        
        # Pass the output through the projection layer
        x = self.proj(x)
        # print('After projection', x.shape)
        
        # Pass the output through the final classification layer
        x = self.fc(x)
        # print('After fc', x.shape)

        return x