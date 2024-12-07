import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from math import sqrt
import random
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


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
        if batch_idx % 50 == 0: print(f"Query shape: {q.shape}, Key shape: {k.shape}, Value shape: {v.shape}")
        assert q.shape[-1] == k.shape[-1], f"Query and Key dimension mismatch: {q.shape[-1]} vs {k.shape[-1]}"
        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.d)
        weights = F.softmax(scores, dim=-1)
        if batch_idx % 50 == 0: print(f"Scores shape: {scores.shape}, Weights shape: {weights.shape}")
        attended = torch.matmul(weights, v)
        attended = self.W_out(attended)  # Project back to the original embedding dimension
        if batch_idx % 50 == 0: print(f"Attended shape after projection: {attended.shape}")
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
        self.embed = nn.Embedding(num_embeddings=num_tokens, embedding_dim=dim, padding_idx=0)
        self.norm = SimpleRMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.embed(x))
        for i, layer in enumerate(self.layers):
            x = layer(x)
            assert x.shape[-1] == self.layers[0].attn.d * 8, f"Shape mismatch in layer {i}: expected {self.layers[0].attn.d * 8}, got {x.shape[-1]}"
        return x

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
        if batch_idx % 50 == 0: print(f"Encoder output shape: {encoder_output.shape}")
        x = self.norm(self.embed(tgt))
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, encoder_output)
            if batch_idx % 50 == 0: print(f"Decoder layer {i} output shape: {x.shape}")
        output = self.output_head(x)
        if batch_idx % 50 == 0: print(f"Output shape: {output.shape}")
        return output

multi30k = load_dataset('bentrevett/multi30k', split='train')
print(f"Dataset example: {multi30k[0]}")  # Debug: Print an example to understand the data structure


tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

class Multi30KDataset(Dataset):
    def __init__(self, dataset, src_lang='en', tgt_lang='de', tokenizer=None, max_len=30):
        self.dataset = dataset
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        print(f"Available keys in dataset: {self.dataset[idx].keys()}")
        src = self.dataset[idx][self.src_lang]
        tgt = self.dataset[idx][self.tgt_lang]
        if self.tokenizer:
            src_tensor = self.tokenizer(src, return_tensors='pt', padding='max_length', max_length=self.max_len, truncation=True)['input_ids'].squeeze()
            tgt_tensor = self.tokenizer(tgt, return_tensors='pt', padding='max_length', max_length=self.max_len, truncation=True)['input_ids'].squeeze()
            print("using the dataset")
        else:
            src_tensor = torch.randint(0, 30000, (self.max_len,))
            tgt_tensor = torch.randint(0, 30000, (self.max_len,))
            print("not using the dataset")
        return src_tensor, tgt_tensor

# Instantiate the dataset and dataloader
dataset = Multi30KDataset(multi30k, tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DifferentialTransformerMT(num_tokens=tokenizer.vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):  # Number of epochs
    model.train()
    total_loss = 0
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch_idx, (src, tgt) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch+1}")
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt)  # Predict the next word
            loss = loss_fn(output.view(-1, output.size(-1)), tgt.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

# Validation and Testing Code
multi30k_valid = load_dataset('bentrevett/multi30k', split='validation')
multi30k_test = load_dataset('bentrevett/multi30k', split='test')

valid_dataset = Multi30KDataset(multi30k_valid, tokenizer=tokenizer)
test_dataset = Multi30KDataset(multi30k_test, tokenizer=tokenizer)

valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Validation Function
def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            for src, tgt in tepoch:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt)
                loss = loss_fn(output.view(-1, output.size(-1)), tgt.reshape(-1))
                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)

# Run validation and testing
eval_loss = evaluate_model(model, valid_loader, loss_fn, device)
print(f"Validation Loss: {eval_loss}")

test_loss = evaluate_model(model, test_loader, loss_fn, device)
print(f"Test Loss: {test_loss}")
