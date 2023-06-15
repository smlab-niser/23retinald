import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from posenc import positionalencoding2d, positionalencoding2db


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, importance_weights, num_patches):
        # Calculate new patch sizes based on importance weights
        average_weight = importance_weights.mean(dim=1)  # Calculate the average weight
        # Define the scaling factor
        scaling_factor = 0.25  # Adjust this value to control the impact of the weights

        # Calculate new patch sizes with scaled weights
        scaled_weights = importance_weights.pow(scaling_factor)
        new_patch_sizes = (32 * average_weight / scaled_weights).clamp(max=32)
        new_patch_sizes = (new_patch_sizes / 32).round() * 32
        new_patch_sizes = new_patch_sizes #.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Reshape to match patch dimensions
        # print("new_patch_sizes: ", new_patch_sizes)

        # Divide patches further based on new patch sizes
        new_patches = []
        subdivided_patch_indices = []  # Store the indices of patches that were subdivided
        
        for i in range(num_patches):
            patch_size = new_patch_sizes[0, i]  # Access the element at index (0, i)
            if (patch_size < 32) and (patch_size > 3):
                subdivided_patch_indices.append(i)
                num_subpatches = int(32 / patch_size)
                subpatch_size = torch.tensor(32 / num_subpatches).item()  # Convert to tensor before using .item()
                subpatches = nn.functional.interpolate(self.patch_embed(x), size=(subpatch_size, subpatch_size), mode='bilinear')
                new_patches.extend(torch.split(subpatches, 1, dim=2))
        
        # print("new_patches: ", len(new_patches))
        # Concatenate the new patches along the patch dimension, excluding the original subdivided patches
        final_patches = []
        for i in range(num_patches):
            if i not in subdivided_patch_indices:
                patch_row = i // int(num_patches**0.5)
                patch_col = i % int(num_patches**0.5)
                new_patches.append(self.patch_embed(x)[:, :, patch_row, patch_col])
        final_patches.extend(new_patches)
        # print("final_patches: ", len(final_patches))

        final_patches = torch.stack(new_patches, dim=2)
        # print("final_patches: ", final_patches.size())

        return final_patches


# class IEViT2(nn.Module):
#     def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, num_layers, dim_feedforward, mlp_dim, backbone):
#         super().__init__()

#         assert img_size % patch_size == 0, 'image size must be divisible by patch size'
#         self.num_patches = (img_size // patch_size) ** 2
#         self.num_layers = num_layers
#         self.embed_dim =  embed_dim

#         self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
#         self.importance_weights = nn.Parameter(torch.ones(1, self.num_patches))  # Learnable importance weights
#         self.importance_weights.data = F.softmax(self.importance_weights, dim=0)  # Normalize importance weights
        
#         # Initialize the TransformerEncoder
#         encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=dim_feedforward)
#         self.transformer = TransformerEncoder(encoder_layer, num_layers)

#         self.layer_norm = nn.LayerNorm(embed_dim)
#         self.backbone = backbone

#         self.mlp_head = nn.Sequential(
#             nn.Linear((self.num_patches + 1 + num_layers) * embed_dim, mlp_dim),
#             nn.GELU(),
#             nn.Linear(mlp_dim, num_classes))
        
#         # Initialize positional embedding
#         self.pos_embed = nn.Parameter(torch.zeros(1, 5000, embed_dim))
#         nn.init.normal_(self.pos_embed, std=0.02)

#     def forward(self, x):

#         ximg = self.backbone(x) 
#         patches = self.patch_embed(x, self.importance_weights, self.num_patches)             
#         patches = patches.flatten(2).transpose(1, 2) 
#         # print(patches.size())
#         cls_tokens = self.cls_token.expand(patches.shape[0], -1, -1) 
#         x = torch.cat((cls_tokens, patches), dim=1)
#         print(x.size())
#         print(self.pos_embed.size())
#         x = x + self.pos_embed[:, :x.size(1), :]
    
#         for i in range(self.num_layers):
#             transformer_layer = self.transformer.layers[i]
#             x = transformer_layer(x)                               
#             x = torch.cat((ximg.unsqueeze(1), x), dim=1)         
        
#         x = self.layer_norm(x) 
#         x = x.flatten(1)  
#         x = self.mlp_head(x)

#         return x


class IEViT21(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, num_layers, dim_feedforward, mlp_dim, backbone):
        super().__init__()

        assert img_size % patch_size == 0, 'image size must be divisible by patch size'
        self.num_patches = (img_size // patch_size) ** 2
        self.num_layers = num_layers
        self.embed_dim =  embed_dim

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.importance_weights = nn.Parameter(torch.ones(1, self.num_patches))  # Learnable importance weights
        self.importance_weights.data = F.softmax(self.importance_weights, dim=0)  # Normalize importance weights
        
        # Initialize the TransformerEncoder
        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=dim_feedforward)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.backbone = backbone

        self.mlp_head = nn.Sequential(
            nn.Linear((self.num_patches + 1 + num_layers) * embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes))
        
        # Initialize positional embedding
        self.pos_embed = nn.Parameter(positionalencoding2d(embed_dim, (img_size // patch_size) + 1, (img_size // patch_size) + 1))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):

        ximg = self.backbone(x) 
        patches = self.patch_embed(x, self.importance_weights, self.num_patches)  
        print(patches.size())
        # Update positional embedding size if the number of patches changes
        if patches.shape[1] != self.pos_embed.shape[2]:
            print("poop")
            self.pos_embed = nn.Parameter(positionalencoding2d(self.pos_embed.shape[0], patches.shape[1] , patches.shape[1]))
            
        patches = patches.flatten(2).transpose(1, 2) 
        # print(patches.size())
        cls_tokens = self.cls_token.expand(patches.shape[0], -1, -1) 
        x = torch.cat((cls_tokens, patches), dim=1)
        
        print(x.size())
        print(self.pos_embed.size())
        
         # Reshape pos_embed to match the desired size
        pos_embed_reshaped = self.pos_embed.view(1, -1, self.embed_dim)
        pos_embed_reshaped = pos_embed_reshaped.transpose(0, 1)
        pos_embed_reshaped = torch.cat([pos_embed_reshaped, torch.zeros(1, 1, self.embed_dim)], dim=1)

        x = x + pos_embed_reshaped[:, :x.size(1), :]
    
        for i in range(self.num_layers):
            transformer_layer = self.transformer.layers[i]
            x = transformer_layer(x)                               
            x = torch.cat((ximg.unsqueeze(1), x), dim=1)         
        
        x = self.layer_norm(x) 
        x = x.flatten(1)  
        x = self.mlp_head(x)

        return x



class IEViT2(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, num_layers, dim_feedforward, mlp_dim, backbone):
        super().__init__()

        assert img_size % patch_size == 0, 'image size must be divisible by patch size'
        self.num_patches = (img_size // patch_size) ** 2
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.importance_weights = nn.Parameter(torch.ones(1, self.num_patches))  # Learnable importance weights
        self.importance_weights.data = F.softmax(self.importance_weights, dim=0)  # Normalize importance weights
        
        # Initialize the TransformerEncoder
        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=dim_feedforward)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.backbone = backbone

        self.mlp_head = nn.Sequential(
            nn.Linear((self.num_patches + 1 + num_layers) * embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes))
        
        # Initialize positional embedding
        self.pos_embed = nn.Parameter(positionalencoding2db(self.num_patches, embed_dim))
        # nn.init.normal_(self.pos_embed.pos_encoding, std=0.02)

    def forward(self, x):
        ximg = self.backbone(x) 
        patches = self.patch_embed(x, self.importance_weights, self.num_patches)
        print(patches.size())
        patches = patches.flatten(2).transpose(1, 2)
        print(patches.size())
        cls_tokens = self.cls_token.expand(patches.shape[0], -1, -1) 
        x = torch.cat((cls_tokens, patches), dim=1)
        print(x.size())
        print(self.pos_embed.size())
        # Apply positional encoding
        x = x + self.pos_embed(x, patches)
    
        for i in range(self.num_layers):
            transformer_layer = self.transformer.layers[i]
            x = transformer_layer(x)                               
            x = torch.cat((ximg.unsqueeze(1), x), dim=1)         
        
        x = self.layer_norm(x) 
        x = x.flatten(1)  
        x = self.mlp_head(x)

        return x

