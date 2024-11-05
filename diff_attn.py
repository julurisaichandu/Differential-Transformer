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


    def forward(self, x, mask = None):
        batch_size , seq_len, _ = x.shape

        # Linear projections ----> multiple heads
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, 2, self.d_head)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, 2, self.d_head)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, -1)

        # Split into two groups
        q1, q2 = q[..., 0, :], q[..., 1, :]
        k1, k2 = k[..., 0, :], k[..., 1, :]

        scale = 1.0 / math.sqrt(self.d_head)
        score_1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        score_2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        # Applying mask as usual
        if mask is not None:
            score_1 = score_1.masked_fill(mask == 0, float('-inf'))
            score_2 = score_2.masked_fill(mask == 0, float('-inf'))


        # Computing Differential Attn

        attn1 = F.softmax(score_1, dim=-1)
        attn2 = F.softmax(score_2, dim=-1)

        lambda_val = self.compute_lambda()
        attn_diff = attn1 - lambda_val * attn2

        out = torch.matmul(attn_diff, v)

        out = self.group_norm(out)
        out = out * (1 - self.lambda_init)

        out = self.o_proj(out.view(batch_size, seq_len, -1))

        return out