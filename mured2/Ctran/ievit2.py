import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from posenc import positionalencoding2d2

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, importance_weights, num_patches):
        # Calculate new patch sizes based on importance weights
        average_weight = importance_weights.mean(dim=1)
        scaling_factor = 0.25

        scaled_weights = importance_weights.pow(scaling_factor)
        new_patch_sizes = (32 * average_weight / scaled_weights).clamp(max=32)
        new_patch_sizes = (new_patch_sizes / 32).round() * 32

        new_patches = []
        subdivided_patch_indices = []

        for i in range(num_patches):
            patch_size = new_patch_sizes[0, i]
            if (patch_size < 32) and (patch_size > 3):
                subdivided_patch_indices.append(i)
                num_subpatches = int(32 / patch_size)
                subpatch_size = torch.tensor(32 / num_subpatches).item()
                subpatches = nn.functional.interpolate(self.patch_embed(x), size=(subpatch_size, subpatch_size), mode='bilinear')
                new_patches.extend(torch.split(subpatches, 1, dim=2))
        
        final_patches = []
        for i in range(num_patches):
            if i not in subdivided_patch_indices:
                patch_row = i // int(num_patches**0.5)
                patch_col = i % int(num_patches**0.5)
                new_patches.append(self.patch_embed(x)[:, :, patch_row, patch_col])
        final_patches.extend(new_patches)

        final_patches = torch.stack(new_patches, dim=2)

        return final_patches


class IEViT2(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, num_layers, dim_feedforward, mlp_dim, backbone):
        super().__init__()

        assert img_size % patch_size == 0, 'image size must be divisible by patch size'
        self.num_patches = (img_size // patch_size) ** 2
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.importance_weights = nn.Parameter(torch.ones(1, self.num_patches))
        self.importance_weights.data = F.softmax(self.importance_weights, dim=1)
        
        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=dim_feedforward)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.backbone = backbone

        self.mlp_head = nn.Sequential(
            nn.Linear((self.num_patches + 1 + num_layers) * embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes))
        
        # Initialize positional embedding
        self.pos_embed = None

    # def generate_positional_encoding(self, num_patches, device):
    #     if self.pos_embed is None or self.pos_embed.shape[2] != num_patches:
    #         embed_dim = self.embed_dim * 2  # Adjust embed_dim to match the desired size
    #         pos_encoding = positionalencoding2d2(embed_dim, num_patches + 1)
    #         self.pos_embed = nn.Parameter(pos_encoding.to(device), requires_grad=False)
    
    def generate_positional_encoding(self, num_patches, device):
        if self.pos_embed is None or self.pos_embed.shape[2] < num_patches:
            embed_dim = self.embed_dim * 2  # Adjust embed_dim to match the desired size
            pos_encoding = positionalencoding2d2(embed_dim, num_patches + 1)

            if self.pos_embed is not None:
                # Resize the positional embedding and preserve existing information
                old_num_patches = self.pos_embed.shape[2]
                new_pos_encoding = torch.zeros_like(pos_encoding)
                new_pos_encoding[:, :, :old_num_patches] = self.pos_embed[:, :, :old_num_patches]
                new_pos_encoding[:, :, old_num_patches:] = pos_encoding[:, :, old_num_patches:]
                pos_encoding = new_pos_encoding

            self.pos_embed = nn.Parameter(pos_encoding.to(device), requires_grad=False)


    def forward(self, x):
        ximg = self.backbone(x) 
        device = ximg.device  # Get the device from ximg
        #print("ximg: ", ximg.size())
        patches = self.patch_embed(x, self.importance_weights, self.num_patches)
        #("patches: ", patches.size())
        
        self.generate_positional_encoding(patches.shape[2], device)  # Pass the device
        #("pos_embed: ", self.pos_embed.size())
        
        pos_embed_reshaped = self.pos_embed.repeat(x.size(0), 1, 1).transpose(1,2)
        #("pos_embed_reshaped: ", pos_embed_reshaped.size())
        
        patches = patches.flatten(2).transpose(1, 2) 
        #("patches reshaped: ", patches.size())
        
        cls_tokens = self.cls_token.expand(patches.shape[0], -1, -1) 
        x = torch.cat((cls_tokens, patches), dim=1)
        #("after csl token: ", x.size())
        
        x = x + pos_embed_reshaped[:, :x.size(1), :]
        #(x.size())
    
        for i in range(self.num_layers):
            transformer_layer = self.transformer.layers[i]
            x = transformer_layer(x)                               
            x = torch.cat((ximg.unsqueeze(1), x), dim=1)         
        
        x = self.layer_norm(x) 
        #("after norm: ", x.size())
        x = x.flatten(1) 
        #("after flatten: ", x.size()) 
        
        # Dynamically adjust input size of linear layer while preserving weights
        linear_input_dim = x.size(1)
        #(linear_input_dim)
        #(self.mlp_head[0].weight.shape[1])
        if linear_input_dim > self.mlp_head[0].weight.shape[1]:
            # Expand the weight matrix if new patches are added
            weight = self.mlp_head[0].weight
            new_weight = torch.zeros(weight.shape[0], linear_input_dim, device=weight.device)
            new_weight[:, :weight.shape[1]] = weight
            self.mlp_head[0].weight.data = new_weight
        elif linear_input_dim < self.mlp_head[0].weight.shape[1]:
            # Slice the weight matrix if fewer patches are present
            weight = self.mlp_head[0].weight
            self.mlp_head[0].weight.data = weight[:, :linear_input_dim]
        
        x = self.mlp_head(x)
        #("after mlp head: ", x.size()) 

        return x
