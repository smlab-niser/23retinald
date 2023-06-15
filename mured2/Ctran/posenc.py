import math
import torch
from torch import nn
from pdb import set_trace as stop


def positionalencoding2d(d_model, height, width):
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    print(pe.size())

    return pe



class positionalencoding2db(nn.Module):
    def __init__(self, num_patches, embed_dim, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, patches, mask):
        assert mask is not None
        not_mask = ~mask

        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.embed_dim, device=patches.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.embed_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos[:, :self.num_patches + 1, :]  # Truncate the positional encoding to match the number of patches

# import math
# import torch
# from torch import nn

# def positionalencoding2d2(embed_dim, num_patches):
#     position = torch.arange(num_patches).unsqueeze(0)  # Shape: (1, num_patches)
#     div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))  # Shape: (embed_dim/2,)
#     div_term = div_term.unsqueeze(0)  # Shape: (1, embed_dim/2)
#     scaled_time = position * div_term  # Shape: (1, num_patches, embed_dim/2)

#     # Calculate sine and cosine components
#     sin_component = torch.sin(scaled_time[:, :, 0::2])  # Shape: (1, num_patches, embed_dim/2)
#     cos_component = torch.cos(scaled_time[:, :, 1::2])  # Shape: (1, num_patches, embed_dim/2)

#     # Combine sine and cosine components
#     pos_encoding = torch.cat([sin_component, cos_component], dim=-1)  # Shape: (1, num_patches, embed_dim)

#     return pos_encoding

import torch
import math

def positionalencoding2d2(embed_dim, num_patches):
    position = torch.arange(num_patches).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_patches)
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))  # Shape: (embed_dim/2,)
    div_term = div_term.unsqueeze(0).unsqueeze(-1)  # Shape: (1, embed_dim/2, 1)
    scaled_time = position * div_term  # Shape: (1, num_patches, embed_dim/2)

    # Calculate sine and cosine components
    sin_component = torch.sin(scaled_time[:, :, 0::2])  # Shape: (1, num_patches, embed_dim/2)
    cos_component = torch.cos(scaled_time[:, :, 1::2])  # Shape: (1, num_patches, embed_dim/2)

    # Combine sine and cosine components
    pos_encoding = torch.cat([sin_component, cos_component], dim=-1)  # Shape: (1, num_patches, embed_dim)

    return pos_encoding #.transpose(1, 2).reshape(1, embed_dim, int(math.sqrt(num_patches)), int(math.sqrt(num_patches)))