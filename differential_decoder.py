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
        return (x / (rms + self.eps)) * self.scale


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
        x = self.dropout(x)
        return self.linear2(x)


class DifferentialAttention(nn.Module):
    """
    Differential Attention module with learnable lambda.
    """

    def __init__(self, d: int, embedding_dim: int, lambda_init: float = 0.8):
        super(DifferentialAttention, self).__init__()
        self.d = d
        self.lambda_param = nn.Parameter(torch.full((d,), lambda_init))
        self.W_q = nn.Linear(embedding_dim, d)
        self.W_k = nn.Linear(embedding_dim, d)
        self.W_v = nn.Linear(embedding_dim, d)
        self.projection = nn.Linear(d, embedding_dim)

    def forward(self, x: Tensor, context: Tensor = None) -> Tensor:
        # If context is None, it's self-attention; otherwise, it's cross-attention
        if context is None:
            context = x

        queries = self.W_q(x)
        keys = self.W_k(context)
        values = self.W_v(context)

        attention_scores = torch.einsum('bqd,bkd->bqk', queries, keys) / sqrt(self.d)
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_values = torch.einsum('bqk,bkd->bqd', attention_weights, values)

        # Applying differential scaling
        weighted_values = self.lambda_param * weighted_values
        return self.projection(weighted_values)


class DifferentialTransformerBlock(nn.Module):
    """
    Implements a differential transformer block for encoder or decoder.
    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.1, lambda_init: float = 0.8,
                 cross_attention: bool = False):
        super(DifferentialTransformerBlock, self).__init__()
        self.attn = DifferentialAttention(d=dim, embedding_dim=dim, lambda_init=lambda_init)
        self.cross_attn = DifferentialAttention(d=dim, embedding_dim=dim,
                                                lambda_init=lambda_init) if cross_attention else None
        self.ffn = FeedForward(dim, dim * 4, dropout)
        self.norm1 = SimpleRMSNorm(dim)
        self.norm2 = SimpleRMSNorm(dim)
        self.norm3 = SimpleRMSNorm(dim) if cross_attention else None

    def forward(self, x: Tensor, encoder_output: Tensor = None) -> Tensor:
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + residual

        if self.cross_attn and encoder_output is not None:
            residual = x
            x = self.norm3(x)
            x = self.cross_attn(x, encoder_output)
            x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        return x + residual


class DifferentialTransformerEncoder(nn.Module):
    """
    Implements the Encoder using differential transformer blocks.
    """

    def __init__(self, dim: int, depth: int, heads: int, dropout: float = 0.1, lambda_init: float = 0.8):
        super(DifferentialTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [DifferentialTransformerBlock(dim, heads, dropout, lambda_init) for _ in range(depth)])
        self.norm = SimpleRMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class DifferentialTransformerDecoder(nn.Module):
    """
    Implements the Decoder using differential transformer blocks.
    """

    def __init__(self, dim: int, depth: int, heads: int, dropout: float = 0.1, lambda_init: float = 0.8):
        super(DifferentialTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [DifferentialTransformerBlock(dim, heads, dropout, lambda_init, cross_attention=True) for _ in
             range(depth)])
        self.norm = SimpleRMSNorm(dim)

    def forward(self, x: Tensor, encoder_output: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output)
        return self.norm(x)


class DifferentialTransformer(nn.Module):
    """
    Implements a full Differential Transformer model with Encoder and Decoder.
    """

    def __init__(self, dim: int = 512, depth: int = 6, heads: int = 8, dropout: float = 0.1, lambda_init: float = 0.8,
                 num_tokens: int = 10000):
        super(DifferentialTransformer, self).__init__()
        self.embedding = nn.Embedding(num_tokens, dim)
        self.encoder = DifferentialTransformerEncoder(dim, depth, heads, dropout, lambda_init)
        self.decoder = DifferentialTransformerDecoder(dim, depth, heads, dropout, lambda_init)
        self.output_head = nn.Linear(dim, num_tokens)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        src_embedded = self.embedding(src)
        tgt_embedded = self.embedding(tgt)

        encoder_output = self.encoder(src_embedded)
        decoder_output = self.decoder(tgt_embedded, encoder_output)

        return self.output_head(decoder_output)


if __name__ == "__main__":
    # Example data: batch of sentences (batch_size=2, sequence_length=5)
    src = torch.randint(0, 10000, (2, 5))  # Source language token IDs
    tgt = torch.randint(0, 10000, (2, 5))  # Target language token IDs

    model = DifferentialTransformer()
    output = model(src, tgt)
    print(f"Output shape: {output.shape}")  # Expected shape: (batch_size, seq_len, vocab_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Example training step
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])  # Shift target by one for teacher forcing
        loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
