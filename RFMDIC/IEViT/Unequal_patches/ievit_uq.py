import torch.nn as nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import sys
sys.path.extend(["..", "../backbone","../.."])
from patch import generate_patches
from clstoken import CLSToken 
from posenc import positionalencoding2d2

class IEViT(nn.Module):
    def __init__(self, img_size, patch_dim, in_channels, num_classes, embed_dim, num_heads, num_layers, 
                 dim_feedforward, mlp_dim, backbone):
        super().__init__()

        self.patches = generate_patches(img_size, patch_dim)
        self.patch_sizes = sorted(self.patches, key=lambda patch: (patch[0], patch[1]))
        self.num_patches = len(self.patch_sizes)
        self.num_layers = num_layers

        self.patch_embed = nn.ModuleList()
        # #print(self.patch_sizes)
        for patch_size in self.patch_sizes:
            top, left, bottom, right = patch_size
            patch_h = bottom - top
            patch_w = right - left
            self.patch_embed.append(nn.Conv2d(in_channels, embed_dim, kernel_size=(patch_h, patch_w), stride=(1, 1), padding=0))
            
        #print(len(self.patch_sizes))
        
        self.pos_embed = nn.Parameter(positionalencoding2d2(embed_dim, self.num_patches)) 
        self.cls_token = CLSToken(in_channels, embed_dim) 

        # Initialize the TransformerEncoder
        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=dim_feedforward)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.backbone = backbone

        self.mlp_head = nn.Sequential(
            nn.Linear((self.num_patches + 1 + num_layers) * embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes))

    def forward(self, x):
        
        #print("Image_size: ", x.size())
        ximg = self.backbone(x)
        #print("CNN embedding: ", ximg.size())
        cls_token = self.cls_token(x)
        
        x_patches = []
        
        for i, patch_size in enumerate(self.patch_sizes):
            patch_embed = self.patch_embed[i]
            top, left, bottom, right = patch_size
            patch_x = patch_embed(x[:, :, top:bottom, left:right])
            x_patches.append(patch_x)
        #print("Patches_size: ", len(x_patches))   
        
        x = torch.cat(x_patches, dim=2)
        #print("Patch_cat: ", x.size())
        
        x = x.flatten(2).transpose(1, 2)
        #print("x flatten: ", x.size())

        x = torch.cat((cls_token, x), dim=1) 
        # print("x + token size: ", x.size())
        
        # Extend pos_embed dimensions
        extended_pos_embed = torch.cat((torch.zeros((1, 1, self.pos_embed.shape[-1]), device=x.device), self.pos_embed), dim=1)
        # print("extend pos embed:", extended_pos_embed.size())
        x = x + extended_pos_embed 
        #print(x.size())

        for i in range(self.num_layers):
            transformer_layer = self.transformer.layers[i]
            x = transformer_layer(x)
            x = torch.cat((ximg.unsqueeze(1), x), dim=1)
            
        #print("After Transformers:")
        x = self.layer_norm(x)
        #print(x.size())
        x = x.flatten(1)
        #print(x.size())
        x = self.mlp_head(x)
        #print(x.size())

        return x