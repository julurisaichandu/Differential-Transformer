import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x):

        if x.size(-1) != self.dim:
            if x.size(-1) < self.dim:

                pad_size = self.dim - x.size(-1)
                x = F.pad(x, (0, pad_size))
            else:

                x = x[..., :self.dim]

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm * self.scale


