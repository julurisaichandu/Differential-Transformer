import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from RMSNorm import RMSNorm

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from RMSNorm import RMSNorm


class DifferentialAttention(nn.Module):
    def __init__(self, d_model, d_head, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads

        # Ensure output dimension matches d_model
        self.q_proj = nn.Linear(d_model, 2 * d_head * n_heads, bias=False)
        self.k_proj = nn.Linear(d_model, 2 * d_head * n_heads, bias=False)
        self.v_proj = nn.Linear(d_model, d_head * n_heads, bias=False)
        self.o_proj = nn.Linear(d_head * n_heads, d_model, bias=False)

        # Initialize norm with correct dimension
        self.norm = RMSNorm(d_model)
        self.lambda_param = nn.Parameter(torch.ones(n_heads) * 0.8)
        self.dropout = nn.Dropout(dropout)

    def compute_lambda(self, batch_size, seq_len):
        # Properly indented compute_lambda method
        return self.lambda_param.view(1, self.n_heads, 1, 1).expand(batch_size, self.n_heads, seq_len, seq_len)

    def forward(self, x, context=None, mask=None):
        batch_size, seq_len, _ = x.shape

        if context is None:
            context = x

        # Project inputs
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, 2, self.d_head)
        k = self.k_proj(context).view(batch_size, context.size(1), self.n_heads, 2, self.d_head)
        v = self.v_proj(context).view(batch_size, context.size(1), self.n_heads, self.d_head)

        # Split queries and keys
        q1, q2 = q[..., 0, :], q[..., 1, :]
        k1, k2 = k[..., 0, :], k[..., 1, :]

        # Reshape for attention computation
        q1, q2 = q1.permute(0, 2, 1, 3), q2.permute(0, 2, 1, 3)
        k1, k2 = k1.permute(0, 2, 1, 3), k2.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.d_head)
        score_1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        score_2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        attn1 = F.softmax(score_1, dim=-1)
        attn2 = F.softmax(score_2, dim=-1)

        # Compute differential attention
        lambda_val = self.compute_lambda(batch_size, seq_len)
        attn_diff = attn1 - lambda_val * attn2

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = mask.expand(batch_size, self.n_heads, seq_len, mask.size(-1))
            attn_diff = attn_diff.masked_fill(mask == float('-inf'), float('-inf'))

        attn_weights = F.softmax(attn_diff, dim=-1)
        out = torch.matmul(attn_weights, v)

        # Reshape and normalize
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, self.n_heads * self.d_head)

        # Apply normalization
        out = self.norm(out)

        # Apply output projection
        out = out * (1 - self.lambda_param.mean())
        out = self.o_proj(out)

        return out
