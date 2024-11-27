import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from math import sqrt
import random


class SimpleRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super(SimpleRMSNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.norm(keepdim=True, dim=-1) / sqrt(self.dim)
        result = (x / (rms + self.eps)) * self.scale
        return result

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class OutputHead(nn.Module):
    def __init__(self, dim: int, vocab_size: int):
        super(OutputHead, self).__init__()
        self.linear = nn.Linear(dim, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        result = self.linear(x)
        return result

class DiffAttn(nn.Module):
    def __init__(self, d: int, embedding_dim: int):
        super(DiffAttn, self).__init__()
        self.d = d
        self.W_q = nn.Linear(embedding_dim, d)
        self.W_k = nn.Linear(embedding_dim, d)
        self.W_v = nn.Linear(embedding_dim, d)
        self.W_out = nn.Linear(d, embedding_dim)  # Projection layer to match the original embedding dimension
        self.lambda_param = nn.Parameter(torch.ones(1, 1, d))

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)
        #print(f"Query shape: {q.shape}, Key shape: {k.shape}, Value shape: {v.shape}")
        assert q.shape[-1] == k.shape[-1], f"Query and Key dimension mismatch: {q.shape[-1]} vs {k.shape[-1]}"
        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.d)
        weights = F.softmax(scores, dim=-1)
        #print(f"Scores shape: {scores.shape}, Weights shape: {weights.shape}")
        attended = torch.matmul(weights, v)
        attended = self.W_out(attended)  # Project back to the original embedding dimension
        #print(f"Attended shape after projection: {attended.shape}")
        return attended

class DifferentialTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1, lambda_init: float = 0.8):
        super(DifferentialTransformerBlock, self).__init__()
        self.attn = DiffAttn(d=dim // heads, embedding_dim=dim)
        self.ffn = FeedForward(dim, dim * 4, dropout)
        self.norm = SimpleRMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        attended = self.attn(self.norm(x), self.norm(x), self.norm(x)) + residual
        assert attended.shape == residual.shape, f"Shape mismatch after attention: {attended.shape} vs {residual.shape}"
        residual = attended
        attended = self.ffn(self.norm(attended)) + residual
        assert attended.shape == residual.shape, f"Shape mismatch after feedforward: {attended.shape} vs {residual.shape}"
        return attended

class DifferentialTransformer(nn.Module):
    def __init__(self, dim: int = 512, heads: int = 8, dropout: float = 0.1, lambda_init: float = 0.8, depth: int = 6, num_tokens: int = 30000):
        super(DifferentialTransformer, self).__init__()
        self.layers = nn.ModuleList([DifferentialTransformerBlock(dim, heads, dropout, lambda_init) for _ in range(depth)])
        self.embed = nn.Embedding(num_embeddings=num_tokens, embedding_dim=dim)
        self.norm = SimpleRMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.embed(x))
        for i, layer in enumerate(self.layers):
            x = layer(x)
            assert x.shape[-1] == self.layers[0].attn.d * 8, f"Shape mismatch in layer {i}: expected {self.layers[0].attn.d * 8}, got {x.shape[-1]}"
        return x

# Implementing the Decoder Block
class DifferentialTransformerDecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1, lambda_init: float = 0.8):
        super(DifferentialTransformerDecoderBlock, self).__init__()
        self.self_attn = DiffAttn(d=dim // heads, embedding_dim=dim)
        self.cross_attn = DiffAttn(d=dim // heads, embedding_dim=dim)
        self.ffn = FeedForward(dim, dim * 4, dropout)
        self.norm = SimpleRMSNorm(dim)

    def forward(self, x: Tensor, encoder_output: Tensor) -> Tensor:
        residual = x
        attended = self.self_attn(self.norm(x), self.norm(x), self.norm(x)) + residual
        assert attended.shape == residual.shape, f"Shape mismatch after self-attention: {attended.shape} vs {residual.shape}"
        residual = attended
        attended = self.cross_attn(self.norm(attended), self.norm(encoder_output), self.norm(encoder_output))
        assert attended.shape[:-1] == residual.shape[:-1], f"Shape mismatch after cross-attention: {attended.shape} vs {residual.shape}"
        attended = attended + residual
        residual = attended
        attended = self.ffn(self.norm(attended)) + residual
        assert attended.shape == residual.shape, f"Shape mismatch after feedforward: {attended.shape} vs {residual.shape}"
        return attended

class DifferentialTransformerMT(nn.Module):
    def __init__(self, dim: int = 512, heads: int = 8, dropout: float = 0.1, lambda_init: float = 0.8, depth: int = 6, num_tokens: int = 30000):
        super(DifferentialTransformerMT, self).__init__()
        self.encoder = DifferentialTransformer(dim, heads, dropout, lambda_init, depth, num_tokens)
        self.decoder_layers = nn.ModuleList([DifferentialTransformerDecoderBlock(dim, heads, dropout, lambda_init) for _ in range(depth)])
        self.embed = nn.Embedding(num_embeddings=num_tokens, embedding_dim=dim)
        self.norm = SimpleRMSNorm(dim)
        self.output_head = OutputHead(dim, vocab_size=num_tokens)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        encoder_output = self.encoder(src)
        print(f"Encoder output shape: {encoder_output.shape}")
        x = self.norm(self.embed(tgt))
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, encoder_output)
            print(f"Decoder layer {i} output shape: {x.shape}")
        output = self.output_head(x)
        print(f"Output shape: {output.shape}")
        return output

# Dataset Class for Multi30K-like Data
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab_size=30000, tgt_vocab_size=30000, max_len=30):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = self.src_sentences[idx]
        tgt = self.tgt_sentences[idx]
        src_tensor = torch.randint(0, self.src_vocab_size, (self.max_len,))
        tgt_tensor = torch.randint(0, self.tgt_vocab_size, (self.max_len,))
        return src_tensor, tgt_tensor

# Generating Sample Data
src_sentences = ["a photo of a cat", "a man riding a horse", "a group of people playing football"] * 100
tgt_sentences = ["ein Foto einer Katze", "ein Mann reitet ein Pferd", "eine Gruppe von Menschen spielt Fußball"] * 100

dataset = TranslationDataset(src_sentences, tgt_sentences)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DifferentialTransformerMT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):  # Number of epochs
    model.train()
    total_loss = 0
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt)  # Predict the next word
        loss = loss_fn(output.view(-1, output.size(-1)), tgt.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}")
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")
