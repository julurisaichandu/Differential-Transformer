import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))


    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim =-1, keepdim = True) + self.eps)

        x_norm = x / rms * self.scale
        return x_norm


