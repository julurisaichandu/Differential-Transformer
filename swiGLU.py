import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class swiGLU(nn.Module):
    def __init__(self, d_model, exp_factor = 8/3):
        super().__init__()
        d_ff = int(d_model * exp_factor)
        self.w1 = nn.Linear(d_model, d_ff, bias = False)
        self.w2 = nn.Linear(d_model, d_ff, bias = False)
        self.w3 = nn.Linear(d_ff, d_model, bias = False)

