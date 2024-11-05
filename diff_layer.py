from diff_attn import *
from swiGLU import *


class DifferentialTransformerLayer(nn.Module):
    def __init__(self, d_model, d_head, n_heads, dropout = 0.1):
        super().__init__()
        self.prenorm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.prenorm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        self.attention = DifferentialAttention(d_model, d_head, n_heads, dropout)
        self.ff_wd = swiGLU(d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask = None):

        residual = x
        x = self.prenorm1(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.prenorm2(x)
        x = self.ff_wd(x)
        x = self.dropout(x)
        x = x + residual

        return x