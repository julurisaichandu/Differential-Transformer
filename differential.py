import torch
import torch.nn.functional as F
from torch import nn, Tensor
from math import sqrt

class SimpleRMSNorm(nn.Module):
    """
    Implements Root Mean Square Layer Normalization.
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super(SimpleRMSNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.norm(keepdim=True, dim=-1) / sqrt(self.dim)
        result = (x / (rms + self.eps)) * self.scale
        #print(f"SimpleRMSNorm output shape: {result.shape}")
        return result


class FeedForward(nn.Module):
    """
    Implements the FeedForward network as used in transformers.
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.linear1(x))
        #print(f"FeedForward linear1 output shape: {x.shape}")
        x = self.dropout(x)
        x = self.linear2(x)
        #print(f"FeedForward linear2 output shape: {x.shape}")
        return x


class OutputHead(nn.Module):
    """
    Implements the output layer for prediction.
    """
    def __init__(self, dim: int, vocab_size: int):
        super(OutputHead, self).__init__()
        self.linear = nn.Linear(dim, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        result = self.linear(x)
        #print(f"OutputHead output shape: {result.shape}")
        return result


class DiffAttn(nn.Module):
    """
    Differential Attention module with learnable lambda.
    """
    def __init__(self, d: int, embedding_dim: int):
        super(DiffAttn, self).__init__()
        self.d = d
        self.W_q = nn.Linear(embedding_dim, 2 * d)
        self.W_k = nn.Linear(embedding_dim, 2 * d)
        self.W_v = nn.Linear(embedding_dim, d)  # Project to d dimensions to match attention output
        self.lambda_ = nn.Parameter(torch.randn(1))  # Scalar learnable lambda
        self.lambda_init = 0.05

    def forward(self, X: Tensor) -> Tensor:
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)
        #print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")

        Q1, Q2 = self.split(Q)
        K1, K2 = self.split(K)
        #print(f"Q1 shape: {Q1.shape}, Q2 shape: {Q2.shape}, K1 shape: {K1.shape}, K2 shape: {K2.shape}")

        s = 1 / sqrt(self.d)
        A1 = (Q1 @ K1.transpose(-1, -2)) * s
        A2 = (Q2 @ K2.transpose(-1, -2)) * s
        #print(f"A1 shape: {A1.shape}, A2 shape: {A2.shape}")

        A1_softmax = F.softmax(A1, dim=-1)
        A2_softmax = F.softmax(A2, dim=-1)
        #print(f"A1_softmax shape: {A1_softmax.shape}, A2_softmax shape: {A2_softmax.shape}")

        lambda_ = torch.exp(self.lambda_) + self.lambda_init
        #print(f"Learnable lambda: {lambda_}")

        # Calculate the differential attention output
        differential_attn = A1_softmax - lambda_ * A2_softmax
        result = torch.bmm(differential_attn, V)
        #print(f"DiffAttn result shape: {result.shape}")
        return result

    @staticmethod
    def split(X: Tensor) -> (Tensor, Tensor):
        half_dim = X.shape[-1] // 2
        return X[..., :half_dim], X[..., half_dim:]


class MultiHeadDifferentialAttention(nn.Module):
    """
    Multi-Head Differential Attention module.
    """
    def __init__(self, h: int, d: int, embedding_dim: int, lambda_init: float):
        super(MultiHeadDifferentialAttention, self).__init__()
        self.h = h
        self.d = d
        self.lambda_init = lambda_init
        self.embedding_dim = embedding_dim
        self.diff_attn_heads = nn.ModuleList([DiffAttn(d, embedding_dim) for _ in range(h)])
        self.W_o = nn.Linear(h * d, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, X: Tensor) -> Tensor:
        O_list = [head(X) for head in self.diff_attn_heads]
        O_concat = torch.cat(O_list, dim=-1)
        #print(f"MultiHead O_concat shape: {O_concat.shape}")
        result = self.W_o(O_concat)
        #print(f"MultiHead W_o output shape: {result.shape}")
        result = self.norm(result)
        #print(f"MultiHead norm output shape: {result.shape}")
        result = result * (1 - self.lambda_init)
        return result


class DifferentialTransformerBlock(nn.Module):
    """
    Implements a Differential Transformer Block.
    """
    def __init__(self, dim: int, heads: int = 12, dropout: float = 0.1, lambda_init: float = 0.05):
        super(DifferentialTransformerBlock, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.lambda_init = lambda_init

        self.attn = MultiHeadDifferentialAttention(heads, dim, dim, lambda_init)
        self.ffn = FeedForward(dim, dim * 4, dropout)
        self.norm = SimpleRMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        attended = self.attn(self.norm(x)) + residual
        #print(f"DifferentialTransformerBlock first attention output shape: {attended.shape}")
        residual_two = attended
        attended = self.attn(self.norm(residual_two)) + residual_two
        #print(f"DifferentialTransformerBlock second attention output shape: {attended.shape}")
        return attended


class DifferentialTransformer(nn.Module):
    """
    Implements a full Differential Transformer.
    """
    def __init__(self, dim: int = 3072, heads: int = 12, dropout: float = 0.1, lambda_init: float = 0.8, depth: int = 24, num_tokens: int = 30000):
        super(DifferentialTransformer, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.lambda_init = lambda_init
        self.depth = depth
        self.num_tokens = num_tokens

        self.layers = nn.ModuleList([DifferentialTransformerBlock(dim, heads, dropout, lambda_init) for _ in range(depth)])
        self.embed = nn.Embedding(num_embeddings=num_tokens, embedding_dim=dim)
        self.norm = SimpleRMSNorm(dim)

    def forward(self, x):
        x = self.norm(self.embed(x))
        #print(f"Embedding output shape: {x.shape}")
        for i, layer in enumerate(self.layers):
            x = layer(x)
            #print(f"Layer {i} output shape: {x.shape}")
        output = OutputHead(self.dim, vocab_size=self.num_tokens)(x)
        ##print(f"Final output shape: {output.shape}")
        return output


# Example usage:
# batch_size, seq_len, embedding_dim, h, lambda_init = 32, 128, 64, 8, 0.05
# x = torch.randint(0, 256, (batch_size, seq_len))
# transformer = DifferentialTransformer(heads=h, dim=embedding_dim, lambda_init=lambda_init)
# output = transformer(x)
# ##print(f"Output shape: {output.shape}")
