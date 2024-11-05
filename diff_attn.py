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

        self.q_proj = nn.Linear(d_model, 2 * d_head * n_heads, bias=False)
        self.k_proj = nn.Linear(d_model, 2 * d_head * n_heads, bias=False)
        self.v_proj = nn.Linear(d_model, d_head * n_heads, bias=False)
        self.o_proj = nn.Linear(d_head * n_heads, d_model, bias=False)

        # Lambda Parameters
        self.lambda_q1 = nn.Parameter(torch.zeros(d_head))
        self.lambda_k1 = nn.Parameter(torch.zeros(d_head))
        self.lambda_q2 = nn.Parameter(torch.zeros(d_head))
        self.lambda_k2 = nn.Parameter(torch.zeros(d_head))
        self.lambda_init = 0.8

        self.dropout = nn.Dropout(dropout)

        self.group_norm = nn.GroupNorm(n_heads, d_head * n_heads)

    def compute_lambda(self):
        lambda_val = (
            torch.exp(self.lambda_q1 * self.lambda_k1) -
            torch.exp(self.lambda_q2 * self.lambda_k2) +
            self.lambda_init
        )
        return lambda_val.view(1, 1, 1, -1)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project the inputs to queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, 2, self.d_head)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, 2, self.d_head)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)

        q1, q2 = q[..., 0, :], q[..., 1, :]
        k1, k2 = k[..., 0, :], k[..., 1, :]

        q1, q2 = q1.permute(0, 2, 1, 3), q2.permute(0, 2, 1, 3)
        k1, k2 = k1.permute(0, 2, 1, 3), k2.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.d_head)
        score_1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        score_2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        attn1 = F.softmax(score_1, dim=-1)
        attn2 = F.softmax(score_2, dim=-1)
        lambda_val = self.compute_lambda()

        lambda_val = lambda_val.expand_as(attn1)
        # Subtract attn2 from attn1 with lambda scaling
        attn_diff = attn1 - lambda_val * attn2

        # Apply attention mask (if provided) after computing attn_diff
        if mask is not None:
            attn_diff = attn_diff.masked_fill(mask == 0, float('-inf'))

        # Perform attention over the values
        attn_weights = F.softmax(attn_diff, dim=-1)
        out = torch.matmul(attn_weights, v)

        # Reshape output
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, self.n_heads * self.d_head)
        out = out.transpose(1, 2)  # [batch_size, n_heads * d_head, seq_len]
        out = self.group_norm(out)
        out = out.transpose(1, 2)  # [batch_size, seq_len, n_heads * d_head]

        out = out * (1 - self.lambda_init)
        out = self.o_proj(out)

        return out
