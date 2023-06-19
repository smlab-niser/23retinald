import torch.nn as nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import sys
sys.path.append("..")
from posenc import positionalencoding2d



# class CTranEncoder(nn.Module):
#     def __init__(self, num_classes, embed_dim, num_layers, num_heads, backbone):
#         super(CTranEncoder, self).__init__()
        
#         # Initialize the backbone network
#         self.backbone = backbone
        
#         # Initialize the TransformerEncoder
#         encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=2048)
#         self.transformer = TransformerEncoder(encoder_layer, num_layers)
       
#         # Initialize the final classification layer
#         self.fc = nn.Linear(embed_dim, num_classes)
        
#         # Initialize the projection layer
#         self.proj = nn.Linear(2*embed_dim, embed_dim)
        
#         # Initialize the batch normalization layer
#         self.bn = nn.BatchNorm1d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
#         # Initialize the positional encoding matrix
#         self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, embed_dim))
#         nn.init.normal_(self.positional_encoding, std=0.02)

        
#     def forward(self, x): 
#         # Pass the input through the backbone network
#         #print("Before backbone:", x.shape)
#         x = self.backbone(x)
#         #print("After backbone:", x.shape)
#         # Add the positional encoding to the input
#         x = x + self.positional_encoding[:, :x.size(1), :]
#         #print("After encoding:", x.shape)
#         # Pass the output through the TransformerEncoder
#         x = self.transformer(x)
#         #print("After transform:", x.shape)
#         # Concatenate the first and last hidden states along the last dimension
#         x = torch.cat((x[0], x[-1]), dim=-1)
#         #print("After cat:", x.shape)
#         # Pass the output through the projection layer
#         x = self.proj(x)
#         #print("After proj:", x.shape)
#         # Add a seq_len dimension to x
#         x = x.unsqueeze(2)
#         #print("After unsqueeze:", x.shape)
#         # Pass the output through the batch normalization layer
#         x = self.bn(x.transpose(1, 2)).transpose(1, 2)
#         #print("After bn:", x.shape)
#         # Remove the seq_len dimension from x
#         x = x.squeeze(2)
#         #print("After squeeze:", x.shape)
#         # Pass the output through the final classification layer
#         x = self.fc(x)
#         #print("After fc:", x.shape)
#         return x
        

# Using Positional Encoding Function
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
        self.positional_encoding = nn.Parameter(positionalencoding2d(embed_dim, height=384, width=384))
        nn.init.normal_(self.positional_encoding, std=0.02)
        # print("posenc size:", self.positional_encoding.size())
        
        # Initialize the linear layer for dimensionality reduction
        self.linear_reduction = nn.Linear(384*384, num_classes)
        
        # Initialize the class token
        # self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # nn.init.normal_(self.class_token, std=0.02)
        
    def forward(self, x): 
        # Pass the input through the backbone network
        # print("Before backbone:", x.shape) torch.Size([batch, channel, height, width])
        x = self.backbone1(x)
        # print("After backbone:", x.shape) torch.Size([num_classes, batch, embed_dim])
        # Add the positional encoding to the input
         # Reduce dimensions and expand positional encoding
        positional_encoding = self.positional_encoding.view(self.embed_dim, -1)
        positional_encoding = positional_encoding.unsqueeze(1).expand(-1, x.size(1), -1).transpose(1, 0)
        # print(positional_encoding.size())
        
        # Apply linear reduction
        positional_encoding = self.linear_reduction(positional_encoding)
        # print("positional encoding size after linear reduction:", positional_encoding.size())

        # Perform element-wise addition
        x = x + positional_encoding.transpose(1,0).transpose(2,0)
        # print("After encoding:", x.shape)  torch.Size([num_classes, batch, embed_dim])
        # Pass the output through the TransformerEncoder
        x = self.transformer(x)
        # print("After transform:", x.shape) torch.Size([num_classes, batch, embed_dim])
        # Concatenate the first and last hidden states along the last dimension
        x = torch.cat((x[0], x[-1]), dim=-1)
        # print("After cat:", x.shape) torch.Size([batch, 2*embed_dim])
        # Pass the output through the projection layer
        x = self.proj(x) 
        # print("After proj:", x.shape) torch.Size([batch, embed_dim])
        # Add a seq_len dimension to x
        x = x.unsqueeze(2)
        # print("After unsqueeze:", x.shape) torch.Size([batch, embed_dim, 1]) 
        # Pass the output through the batch normalization layer
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        # print("After bn:", x.shape) torch.Size([batch, embed_dim, 1])
        # Remove the seq_len dimension from x
        x = x.squeeze(2)
        # print("After squeeze:", x.shape) torch.Size([batch, embed_dim])
        # Pass the output through the final classification layer
        x = self.fc(x)
        # print("After fc:", x.shape) torch.Size([batch, num_classes])
        return x  
    
