import torch.nn as nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

def get_dimensions(lst):
    dimensions = []
    while isinstance(lst, list):
        dimensions.append(len(lst))
        lst = lst[0] if lst else None
    return dimensions

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.patch_embed(x)



class IEViT3(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, num_layers,
                 dim_feedforward, mlp_dim, backbone):
        super().__init__()
        self.img_size=img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.num_layers = num_layers

        # self.patch_embed = PatchEmbedding(in_channels, embed_dim, self.patch_size)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Initialize the TransformerEncoder
        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=dim_feedforward)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.backbone = backbone

        self.mlp_head = nn.Sequential(
            nn.Linear((self.num_patches + 1 + num_layers) * embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes))

        # Learnable parameters for patch decomposition
        self.decompose_weights = nn.Parameter(torch.ones(num_layers, dtype=torch.float32))
        print(self.decompose_weights.size())

        # Learnable importance weights for each patch
        self.importance_weights = nn.Parameter(torch.ones(self.num_patches, dtype=torch.float32))
        self.importance_weights.data = F.softmax(self.importance_weights, dim=0)  # Normalize importance weights
        
        # Threshold for patch decomposition
        self.decompose_threshold = 0.2

    def forward(self, x):
        print("x: ", x.size())
        ximg = self.backbone(x)
        print("ximg: ", ximg.size())
        batch_size = x.shape[0]
        print("batch_Size: ", batch_size)

        patch_embed = self.patch_embed(x)
        print("patch embed: ", patch_embed.size())
        
        _, _, patch_height, patch_width = patch_embed.size()
        patch_height = patch_width = int(self.img_size/patch_height)
        print("patch_h, patch_w: ", patch_height, patch_width)

        # Patch decomposition
        patch_sizes = [(patch_height, patch_width)]  # Initialize with the original patch size
        print("patch_Sizes", patch_sizes)
        
        for layer_idx in range(self.num_layers):
            decompose_weight = torch.sigmoid(self.decompose_weights[layer_idx])
            decomposed_size = (int(patch_height * decompose_weight), int(patch_width * decompose_weight))

            # Ensure decomposed patch size is a factor of 32
            decomposed_size = (int(decomposed_size[0] // 32) * 32, int(decomposed_size[1] // 32) * 32)
            print(decomposed_size)

            # Add constraint: do nothing if the decomposed_size < 4
            if decomposed_size[0] >= 4 and decomposed_size[1] >= 4:
                patch_sizes.append(decomposed_size)
                patch_height, patch_width = decomposed_size
                print("patch_h, patch_w: ", patch_height, patch_width)
        
        print("patch_sizes: ", patch_sizes) 
          
        # Generate patches
        patches = []
        for size in patch_sizes:
            ph, pw = size
            # h_step = patch_height // ph
            # w_step = patch_width // pw
            for i in range(0, patch_height, ph):
                for j in range(0, patch_width, pw):
                    patch = (i, j, i + ph, j + pw)
                    patches.append(patch)

        self.num_patches = len(patches)
        print("num_patches: ", self.num_patches)

        x_patches = []
        for patch in patches:
            patch_x = patch_embed[:, :, patch[0]:patch[2], patch[1]:patch[3]]
            x_patches.append(patch_x)
        print("x_patch: ", get_dimensions(x_patches))
        
        x = torch.cat(x_patches, dim=2)
        print("x in patches: ", x.size())
        x = x.flatten(2).transpose(1, 2)
        print("flatten x: ", x.size())

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        print("with cls token x: ", x.size())
        x = x + self.pos_embed
        print(" with pos embed x: ", x.size())

        for i in range(self.num_layers):
            transformer_layer = self.transformer.layers[i]
            x = transformer_layer(x)
            decompose_weights = self.decompose_weights[i].unsqueeze(0).unsqueeze(2)
            importance_weights = self.importance_weights.unsqueeze(0).unsqueeze(1)
            prod_weights = decompose_weights * importance_weights
            mask = (prod_weights > self.decompose_threshold).float()
            x = x * mask + x_patches[i] * (1 - mask)
            x = torch.cat((ximg.unsqueeze(1), x), dim=1)

        # Apply layer normalization
        x = self.layer_norm(x)
        print("After norm: ", x.size())

        # Flatten and pass through MLP head
        x = x.flatten(2).transpose(1, 2)
        print("After flatten: ", x.size())

        # Forward pass through MLP head
        x = self.mlp_head(x)
        print(x.size())

        return x
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class IEViT2(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, num_layers, 
                 dim_feedforward, mlp_dim, backbone):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2 
        self.num_layers = num_layers 

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Initialize the TransformerEncoder
        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=dim_feedforward)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        self.layer_norm = nn.LayerNorm(embed_dim) 
        self.backbone = backbone

        self.mlp_head = nn.Sequential(
            nn.Linear((self.num_patches + 1 + num_layers) * embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes))

        # Learnable parameters for patch decomposition
        self.decompose_weights = nn.Parameter(torch.ones(num_layers, dtype=torch.float32))

    def forward(self, x):
        ximg = self.backbone(x)
        batch_size = x.shape[0]

        patch_embed = self.patch_embed(x)
        _, _, patch_height, patch_width = patch_embed.size()

        # Patch decomposition
        patch_sizes = [(patch_height, patch_width)]  # Initialize with the original patch size
        for layer_idx in range(self.num_layers):
            decompose_weight = torch.sigmoid(self.decompose_weights[layer_idx])
            decomposed_size = (int(patch_height * decompose_weight), int(patch_width * decompose_weight))
    
            # Ensure decomposed patch size is a factor of 32
            decomposed_size = (int(decomposed_size[0] // 32) * 32, int(decomposed_size[1] // 32) * 32)
    
            # Add constraint: do nothing if the decomposed_size < 4
            if decomposed_size[0] >= 4 and decomposed_size[1] >= 4:
                patch_sizes.append(decomposed_size)
                patch_height, patch_width = decomposed_size

        # Generate patches
        patches = []
        for size in patch_sizes:
            ph, pw = size
            h_step = patch_height // ph
            w_step = patch_width // pw
            for i in range(0, patch_height, h_step):
                for j in range(0, patch_width, w_step):
                    patch = (i, j, i + ph, j + pw)
                    patches.append(patch)

        self.num_patches = len(patches)

        x_patches = []
        for patch in patches:
            patch_x = patch_embed[:, :, patch[0]:patch[2], patch[1]:patch[3]]
            x_patches.append(patch_x)

        x = torch.cat(x_patches, dim=2)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed 

        for i in range(self.num_layers):
            transformer_layer = self.transformer.layers[i]
            x = transformer_layer(x)

        # Apply layer normalization
        x = self.layer_norm(x)

        # Concatenate the CNN embedding with the patch embeddings
        ximg = ximg.view(batch_size, -1)
        x = torch.cat((ximg, x), dim=1)

        # Forward pass through MLP head
        x = self.mlp_head(x)

        return x

