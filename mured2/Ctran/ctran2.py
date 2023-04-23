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
        
        # Initialize the batch normalization layer
        self.bn = nn.BatchNorm1d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # Initialize the positional encoding matrix
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, embed_dim))
        nn.init.normal_(self.positional_encoding, std=0.02)

        
    def forward(self, x): 
        # Pass the input through the backbone network
        #print("Before backbone:", x.shape)
        x = self.backbone(x)
        #print("After backbone:", x.shape)
        # Add the positional encoding to the input
        x = x + self.positional_encoding[:, :x.size(1), :]
        #print("After encoding:", x.shape)
        # Pass the output through the TransformerEncoder
        x = self.transformer(x)
        #print("After transform:", x.shape)
        # Concatenate the first and last hidden states along the last dimension
        x = torch.cat((x[0], x[-1]), dim=-1)
        #print("After cat:", x.shape)
        # Pass the output through the projection layer
        x = self.proj(x)
        #print("After proj:", x.shape)
        # Add a seq_len dimension to x
        x = x.unsqueeze(2)
        #print("After unsqueeze:", x.shape)
        # Pass the output through the batch normalization layer
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        #print("After bn:", x.shape)
        # Remove the seq_len dimension from x
        x = x.squeeze(2)
        #print("After squeeze:", x.shape)
        # Pass the output through the final classification layer
        x = self.fc(x)
        #print("After fc:", x.shape)
        return x
        
    
    
