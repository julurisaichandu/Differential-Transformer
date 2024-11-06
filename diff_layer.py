from diff_attn import *
from swiGLU import *


class DifferentialTransformerLayer(nn.Module):
    def __init__(self, d_model, d_head, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.attention = DifferentialAttention(d_model, d_head, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, context=None, mask=None):
        attn_output = self.attention(x, context=context, mask=mask)
        attn_output = self.dropout(attn_output)
        x = x + attn_output
        x = self.norm(x)
        return x
