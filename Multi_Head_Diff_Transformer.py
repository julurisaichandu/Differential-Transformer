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
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.swiGLU = swiGLU(d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.swiGLU(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_head, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.prenorm1 = nn.LayerNorm(d_model)
        self.attention = DifferentialTransformerLayer(d_model, d_head, n_heads, dropout)
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
        self.cross_attention = DifferentialTransformerLayer(d_model, d_head, n_heads, dropout)
        self.prenorm3 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

    def forward(self, x, encoder_out, target_mask=None):
        # Self-attention over the target tokens with masking
        x = x + self.self_attention(self.prenorm1(x), mask=target_mask)
        # Cross-attention over the encoder output
        x = x + self.cross_attention(self.prenorm2(x), encoder_out)
        # Feed-forward network
        x = x + self.ffn(self.prenorm3(x))
        return x


class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, d_head, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, d_head, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, d_head, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, d_head, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

    def forward(self, x, encoder_output, target_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, target_mask=target_mask)
        return x


class EncoderDecoderTransformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model=512,
                 n_heads=8,
                 n_layers=6,
                 d_head=64,
                 d_ff=2048,
                 max_seq_len=512,
                 dropout=0.1):
        super().__init__()

        self.embedding_layer = EmbeddingLayer(vocab_size, d_model, max_seq_len, dropout)
        self.encoder = Encoder(n_layers, d_model, d_head, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, d_head, n_heads, d_ff, dropout)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src_tokens, tgt_tokens):
        # Embedding
        src_embeddings = self.embedding_layer(src_tokens)
        tgt_embeddings = self.embedding_layer(tgt_tokens)

        # Encoder
        encoder_output = self.encoder(src_embeddings)

        # Create target mask to prevent attending to future tokens
        tgt_seq_len = tgt_tokens.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt_tokens.device)

        # Decoder
        decoder_output = self.decoder(tgt_embeddings, encoder_output, target_mask=tgt_mask)

        return decoder_output


# Sample usage
if __name__ == "__main__":
    vocab_size = 10000
    d_model = 512
    n_heads = 8
    n_layers = 6
    d_head = 64
    d_ff = 2048
    max_seq_len = 128
    dropout = 0.1

    model = EncoderDecoderTransformer(vocab_size, d_model, n_heads, n_layers, d_head, d_ff, max_seq_len, dropout)
    src_tokens = torch.randint(0, vocab_size, (2, max_seq_len))
    tgt_tokens = torch.randint(0, vocab_size, (2, max_seq_len))
    output = model(src_tokens, tgt_tokens)
    print(output.shape)  # Expected output: (batch_size, seq_len, d_model)
