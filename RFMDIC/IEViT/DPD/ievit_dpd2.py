import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import sys
sys.path.extend(["..", "../backbone","../.."])
import torch.nn.functional as F
from posenc import positionalencoding2d2
from clstoken import CLSToken 

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, importance_weights, num_patches):
        # Calculate new patch sizes based on importance weights
        average_weight = importance_weights.mean(dim=1)
        scaling_factor = 0.05
        ps = self.patch_size

        scaled_weights = importance_weights.pow(scaling_factor)
        new_patch_sizes = (ps * average_weight.pow(scaling_factor) / scaled_weights).clamp(max=ps)
        new_patch_sizes = [min([32, 16, 8, 4], key=lambda x: abs(x - num)) for num in new_patch_sizes]

        new_patches = []
        subdivided_patch_indices = []
        subpatch_counts = []
        H = x.shape[3]
        # print(new_patch_sizes.size())

        for i in range(num_patches):
            patch_size = new_patch_sizes[0,i]
            if (patch_size < ps) and (patch_size > 3):
                subdivided_patch_indices.append(i)
                num_subpatches = int(ps / patch_size) # In row or column
                subpatch_size = int(ps / num_subpatches)
                patches = []
                for j in range(num_subpatches):
                    ini_col = (i * ps) % H
                    ini_row = (i * ps * ps) // H
                    start_col = ini_col + ((j * subpatch_size) % (num_subpatches * subpatch_size))
                    start_row = ini_row + ((j * subpatch_size) // num_subpatches)
                    end_row = start_row + subpatch_size
                    end_col = start_col + subpatch_size
                    patch = x[:, :, start_row:end_row, start_col:end_col]
                    patches.append(self.patch_embed(patch))
                new_patches.extend(patches)
                subpatch_counts.append(num_subpatches)
            else:
                subpatch_counts.append(1)

        final_patches = []
        patch_index = 0

        for i in range(num_patches):
            if i in subdivided_patch_indices:
                num_subpatches = subpatch_counts[i]
                final_patches.extend(new_patches[patch_index:patch_index + num_subpatches])
                patch_index += num_subpatches
            else:
                ini_col = (i * ps) % H
                ini_row = (i * ps * ps) // H
                end_row = ini_col + ps
                end_col = ini_col + ps
                final_patches.append(self.patch_embed(x)[:, :, ini_row:end_row, ini_col:end_col])

        final_patches = torch.stack(final_patches, dim=2)
        # if final_patches.size(2) != 144: print()

        return final_patches, subdivided_patch_indices



class IEViT(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, num_layers, dim_feedforward, mlp_dim, backbone):
        super().__init__()

        assert img_size % patch_size == 0, 'image size must be divisible by patch size'
        self.num_patches = (img_size // patch_size) ** 2
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = CLSToken(in_channels, embed_dim) 
        
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
    
    def generate_positional_encoding(self, subdivided_patch_indices, device):
        embed_dim = self.embed_dim   # Adjust embed_dim to match the desired size
        num_patches = self.num_patches + len(subdivided_patch_indices)
        pos_encoding = positionalencoding2d2(embed_dim, num_patches)
        
        if self.pos_embed is not None:
            # Copy the existing positional embeddings for non-subdivided patches
            new_pos_encoding = self.pos_embed.clone()
            
            # Generate new positional embeddings for subdivided patches
            for i, patch_index in enumerate(subdivided_patch_indices):
                new_pos_encoding[:, patch_index] = pos_encoding[:, self.num_patches + i]
            pos_encoding = new_pos_encoding
            
        self.pos_embed = nn.Parameter(pos_encoding.to(device), requires_grad=False)


    def forward(self, x):
        ximg = self.backbone(x) # Output.shape = [batch, embed_dim]
        device = ximg.device  # Get the device from ximg
        #print("ximg: ", ximg.size())
        patches, subdivided_patch_indices = self.patch_embed(x, self.importance_weights, self.num_patches) # Output.shape= [batch, N, embed_dim]
        #("patches: ", patches.size())
        
        self.generate_positional_encoding(subdivided_patch_indices, device)  # Pass the device
        #("pos_embed: ", self.pos_embed.size()) # Output.shape = [1, N, embed_dim]
        patches2 = patches.transpose(1,2)
        
        cls_token = self.cls_token(x)
        x = torch.cat((cls_token, patches2), dim=1)
        # cls_tokens = self.cls_token.expand(patches2.shape[0], -1, -1) 
        # x = torch.cat((cls_tokens, patches2), dim=1) # Output.shape = [batch, N+1, embed_dim]
        #("after csl token: ", x.size())

        # Extend pos_embed dimensions
        extended_pos_embed = torch.cat((torch.zeros((1, 1, self.pos_embed.shape[-1]), device=x.device), self.pos_embed), dim=1) # [1, N+1, embed_dim]
        # print("extend pos embed:", extended_pos_embed.size())
        x = x + extended_pos_embed 
        # print("x size: ", x.size())
    
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

        if linear_input_dim > self.mlp_head[0].weight.shape[1]:
            # Expand the weight matrix if new patches are added
            weight = self.mlp_head[0].weight
            new_weight = torch.zeros(weight.shape[0], linear_input_dim, device=weight.device)
            new_weight[:, :weight.shape[1]] = weight

            # Adjust the values of the new parameters
            num_new_patches = linear_input_dim - weight.shape[1]
            new_weight[:, weight.shape[1]:] = weight[:, :1].repeat(1, num_new_patches) / num_new_patches

            self.mlp_head[0].weight.data = new_weight

        elif linear_input_dim < self.mlp_head[0].weight.shape[1]:
            # Slice the weight matrix if fewer patches are present
            weight = self.mlp_head[0].weight
            self.mlp_head[0].weight.data = weight[:, :linear_input_dim]
    
        x = self.mlp_head(x)
        # print("after mlp head: ", x.size()) 

        return x



class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, importance_weights, num_patches):
        # Calculate new patch sizes based on importance weights
        average_weight = importance_weights.mean(dim=1)
        scaling_factor = 0.05
        ps = self.patch_size

        scaled_weights = importance_weights.pow(scaling_factor)
        new_patch_sizes = (ps * average_weight.pow(scaling_factor) / scaled_weights).clamp(max=ps)
        new_patch_sizes = [min([32, 16, 8, 4], key=lambda x: abs(x - num)) for num in new_patch_sizes]

        new_patches = []
        subdivided_patch_indices = []
        subpatch_counts = []
        H = x.shape[3]
        # print(new_patch_sizes.size())

        for i in range(num_patches):
            patch_size = new_patch_sizes[0,i]
            
            if (patch_size < ps) and (patch_size > 3):
                
                subdivided_patch_indices.append(i)
                num_subpatches = int(ps / patch_size) # In row or column
                subpatch_size = int(ps / num_subpatches)
                patches = []
                
                for j in range(num_subpatches):
                    
                    ini_col = (i * ps) % H
                    ini_row = (i * ps * ps) // H
                    start_col = ini_col + ((j * subpatch_size) % (num_subpatches * subpatch_size))
                    start_row = ini_row + ((j * subpatch_size) // num_subpatches)
                    end_row = start_row + subpatch_size
                    end_col = start_col + subpatch_size
                    patch = x[:, :, start_row:end_row, start_col:end_col]
                    
                    
                    kernel_divisor = ps // patch_size # Calculate the number of times the kernel should be divided in each dimension  
                    original_weights = self.patch_embed.weight # Create a new view of the original kernel 
                    modified_weights = original_weights.view(self.embed_dim, self.in_channels, 
                            kernel_divisor, kernel_divisor, patch_size, patch_size) # Reshape the kernel to match the new patch size
                    
                    modified_weights = modified_weights.sum(dim=2).sum(dim=2)  # Sum over the dimensions where the kernel was divided
                    conv_layer = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=patch_size, 
                                           stride=patch_size, bias=self.patch_embed.bias)  # Create a new convolutional layer with the modified kernel size
                    
                    conv_layer.weight = nn.Parameter(modified_weights) # Assign the modified weights to the new convolutional layer
                    embed = conv_layer(patch) # Apply convolution with dynamically adjusted kernel size
                    embed = embed.flatten(2).transpose(1, 2)
                    patches.append(embed)
                    
                new_patches.extend(patches)
                subpatch_counts.append(num_subpatches)
            else:
                subpatch_counts.append(1)

        final_patches = torch.empty(0)
        patch_index = 0

        for i in range(num_patches):
            if i in subdivided_patch_indices:
                num_subpatches = subpatch_counts[i]
                final_patches = torch.cat((final_patches, *new_patches[patch_index:patch_index + num_subpatches]))
                patch_index += num_subpatches
            else:
                ini_col = (i * ps) % H
                ini_row = (i * ps * ps) // H
                end_row = ini_col + ps
                end_col = ini_col + ps
                embed = self.patch_embed(x[:, :, ini_row:end_row, ini_col:end_col])
                embed = embed.flatten(2).transpose(1, 2)
                final_patches = torch.cat((final_patches, embed))

        return final_patches, subdivided_patch_indices