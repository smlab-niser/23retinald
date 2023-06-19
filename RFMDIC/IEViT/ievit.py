import torch.nn as nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer 
import sys
sys.path.extend(["..", "../backbone","../.."])
from clstoken import CLSToken 
from posenc import positionalencoding2d2


class IEViT(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, num_layers, dim_feedforward, mlp_dim, backbone):
        super().__init__()

        assert img_size % patch_size == 0, 'image size must be divisible by patch size'
        self.num_patches = (img_size // patch_size) ** 2 
        self.num_layers = num_layers 
        
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size) 
        self.pos_embed = nn.Parameter(positionalencoding2d2(embed_dim, self.num_patches)) 
        self.cls_token = CLSToken(in_channels, embed_dim) 
 
        # Initialize the TransformerEncoder
        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward = dim_feedforward) 
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        self.layer_norm = nn.LayerNorm(embed_dim) 
        self.backbone = backbone
        
        self.mlp_head = nn.Sequential(
            nn.Linear((self.num_patches + 1 + num_layers) * embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes))

    def forward(self, x):

        ximg = self.backbone(x) 
        # print("ximg size: ", ximg.size())
        
        cls_token = self.cls_token(x)
        # print("cls token:", cls_token.size())
        
        x = self.patch_embed(x)  
        # print("x patch size: ", x.size())
        
        x = x.flatten(2).transpose(1, 2) 
        # print("x flattened size: ", x.size())
        # cls_tokens = self.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls_token, x), dim=1) 
        # print("x + token size: ", x.size())
        
        # Extend pos_embed dimensions
        extended_pos_embed = torch.cat((torch.zeros((1, 1, self.pos_embed.shape[-1]), device=x.device), self.pos_embed), dim=1)
        # print("extend pos embed:", extended_pos_embed.size())
        x = x + extended_pos_embed 
        # print("x size: ", x.size())
        
        for i in range(self.num_layers):
            transformer_layer = self.transformer.layers[i]
            x = transformer_layer(x)                               
            x = torch.cat((ximg.unsqueeze(1), x), dim=1)         
                
        x = self.layer_norm(x) 
        x = x.flatten(1)  
        x = self.mlp_head(x) 
        
        return x





# class IEViT(nn.Module):
#     def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, num_layers, mlp_dim, backbone):
#         super().__init__()

#         assert img_size % patch_size == 0, 'image size must be divisible by patch size'
#         self.num_patches = (img_size // patch_size) ** 2  
#         self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size) # Output.shape= [batch, embed_dim, H, W]

#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches+1, embed_dim))  # Output.shape= [1, num_patches+1, embed_dim]
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # Output.shape= [1, 1, embed_dim]
 
#         # Initialize the TransformerEncoder
#         encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward = mlp_dim) # mlp_dim = dim_feedforward = 2048
#         self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
#         self.layer_norm = nn.LayerNorm(embed_dim) #Nomalizes about the embed_dim dimension index
#         self.backbone = backbone
        
#          # Add an MLP head which is a 2 layers fc layer
#         self.mlp_head = nn.Sequential(
#             nn.Linear((self.num_patches + 1 + num_layers) * embed_dim, mlp_dim),
#             nn.GELU(),
#             nn.Linear(mlp_dim, num_classes))

#     def forward(self, x):

#         ximg = self.backbone(x) # Output.shape = [batch, embed_dim]
#         x = self.patch_embed(x)  # Output.shape= [batch, embed_dim, H, W]
#         x = x.flatten(2).transpose(1, 2) # Output.shape= [batch, N, embed_dim]
#         cls_tokens = self.cls_token.expand(x.shape[0], -1, -1) # Output.shape= [batch, 1, embed_dim], "-1" indicates that dimension is to be left intact
#         x = torch.cat((cls_tokens, x), dim=1) # Output.shape= [batch, N+1, embed_dim]
#         x = x + self.pos_embed # Output.shape = [batch, N+1, embed_dim], this is element wise addition across dim 0 and not concatenation, that's why the dimensions remain same
        
#         outputs = [x] #List to store outputs
        
#         for transformer in self.transformer:
#             x = transformer(x)                                   # Output.shape = [batch, N + 1 + layer_index, embed_dim]
#             x = torch.cat((ximg.unsqueeze(1), x), dim=1)         # Output.shape = [batch, N + 2 + layer_index, embed_dim]
#             outputs.append(x)
            
#         x = outputs[-1]        # Output.shape = [batch, N + 1 + num_layers, embed_dim]
#         x = self.layer_norm(x) # Output.shape = [batch, N + 1 + num_layers, embed_dim]
        
#         # Flatten and pass through MLP head
#         x = x.flatten(1)  # Output.shape = [batch, (N + 1 + num_layers)*embed_dim], "1" means flatten from 1 dimension index onwards
#         x = self.mlp_head(x)  # Output.shape = [batch, num_classes]
        
#         return x