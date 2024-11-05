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
        self.lambda_k1 = nn.Parameter(torch.zeros(d_head))

        self.lambda_q2 = nn.Parameter(torch.zeros(d_head))
        self.lambda_k2 = nn.Parameter(torch.zeros(d_head))

        self.lambda_init = 0.8

        self.dropout = nn.Dropout(dropout)
        self.group_norm = nn.GroupNorm(n_heads, d_head * n_heads)


    def compute_lambda(self):
        lambda_val = (torch.exp(self.lambda_q1 * self.lambda_k1) -
                     torch.exp(self.lambda_q2 * self.lambda_k2) +
                     self.lambda_init)

        return lambda_val
