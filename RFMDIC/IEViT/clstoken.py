
import torch
import torch.nn as nn

class CLSToken(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)
        x = x.unsqueeze(1)

        return x