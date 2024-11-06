import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_layer import DifferentialTransformerLayer
from swiGLU import swiGLU


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, dtype = torch.long, device=x.device)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.swiGLU = swiGLU(d_model)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.swiGLU(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_head, n_heads, d_ff, dropout= 0.1):
        super().__init__()
        self.prenorm1 = nn.LayerNorm(d_model)
        self.attention = DifferentialTransformerLayer(d_model, d_head, n_heads,dropout)
        self.prenorm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attention(self.prenorm1(x))
        x = x + self.ffn(self.prenorm2(x))

        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_head, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.prenorm1 = nn.LayerNorm(d_model)
        self.self_attention = DifferentialTransformerLayer(d_model, d_head, n_heads, dropout)
        self.prenorm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

    def forward(self, x, encoder_out, target_mask=None):
        x = x + self.self_attention(self.prenorm1(x), mask = target_mask)
        # Added Cross Attention here
        x = x + self.cross_attention(self.prenorm2(x), encoder_out)
        x = x + self.ffn(self.prenorm3(x))

        return x

class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, d_head, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(d_model, d_head, n_heads, d_ff, dropout) for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, d_head, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderBlock(d_model,d_head, n_heads, d_ff, dropout) for _ in range(n_layers)

            ]
        )

    def forward(self, x, encoder_output, target_mask = None):
        for layer in self.layers:
            x = layer(x, encoder_output, target_mask=target_mask)
        return x