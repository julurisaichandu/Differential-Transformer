import torch
import torch.nn as nn
import torch.nn.functional as F
from RMSNorm import RMSNorm
from diff_layer import DifferentialTransformerLayer
from swiGLU import swiGLU

class DecoderOnlyModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_head, n_layers, max_seq_len, dropout = 0.1):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer_layers = nn.ModuleList(
            [DifferentialTransformerLayer(d_model, d_head, n_heads, dropout) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask= None):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, dtype= torch.long, device=x.device)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)
        for layer in self.transformer_layers:
            x = layer(x, mask = mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

