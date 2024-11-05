import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DifferentialAttention(nn.Module):
    def __init__(self, d_model, d_head, n_heads, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads

        self.q_proj = nn.Linear(d_model, 2 * d_head * n_heads, bias =False)
        self.k_proj = nn.Linear(d_model, 2 * d_head * n_heads, bias =False)
        self.v_proj = nn.Linear(d_model, 2 * d_head * n_heads, bias =False)

        self.o_proj = nn.Linear(d_head * n_heads, d_model, bias =False)

        # introducing the Lamda Parameters

        self.lambda_q1 = nn.Parameter(torch.zeros(d_head))
