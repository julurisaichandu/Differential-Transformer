import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import Tokenizer
from pathlib import Path
from tqdm import tqdm
import warnings
from typing import Any
import math
from datasets import load_dataset

# Differential Attention Mechanism Implementation
class SimpleRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super(SimpleRMSNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        print(f"[SimpleRMSNorm] Input shape: {x.shape}")
        rms = x.norm(keepdim=True, dim=-1) / torch.sqrt(torch.tensor(self.dim, dtype=torch.float32))
        result = (x / (rms + self.eps)) * self.scale
        print(f"[SimpleRMSNorm] Output shape: {result.shape}")
        return result


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        print(f"[FeedForward] Input shape: {x.shape}")
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        print(f"[FeedForward] Output shape: {x.shape}")
        return x


class DiffAttn(nn.Module):
    def __init__(self, d: int, embedding_dim: int):
        super(DiffAttn, self).__init__()
        self.d = d
        self.W_q = nn.Linear(embedding_dim, 2 * d)
        self.W_k = nn.Linear(embedding_dim, 2 * d)
        self.W_v = nn.Linear(embedding_dim, d)
        self.lambda_ = nn.Parameter(torch.randn(1))
        self.lambda_init = 0.05

    def forward(self, X: Tensor) -> Tensor:
        print(f"[DiffAttn] Input shape: {X.shape}")
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)
        Q1, Q2 = self.split(Q)
        K1, K2 = self.split(K)
        s = 1 / math.sqrt(self.d)
        A1 = (Q1 @ K1.transpose(-1, -2)) * s
        A2 = (Q2 @ K2.transpose(-1, -2)) * s
        A1_softmax = F.softmax(A1, dim=-1)
        A2_softmax = F.softmax(A2, dim=-1)
        lambda_ = torch.exp(self.lambda_) + self.lambda_init
        differential_attn = A1_softmax - lambda_ * A2_softmax
        result = torch.bmm(differential_attn, V)
        print(f"[DiffAttn] Output shape: {result.shape}")
        return result

    @staticmethod
    def split(X: Tensor) -> (Tensor, Tensor):
        half_dim = X.shape[-1] // 2
        return X[..., :half_dim], X[..., half_dim:]


class MultiHeadDifferentialAttention(nn.Module):
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
        print(f"[MultiHeadDifferentialAttention] Input shape: {X.shape}")
        O_list = [head(X) for head in self.diff_attn_heads]
        O_concat = torch.cat(O_list, dim=-1)
        result = self.W_o(O_concat)
        result = self.norm(result)
        result = result * (1 - self.lambda_init)
        print(f"[MultiHeadDifferentialAttention] Output shape: {result.shape}")
        return result


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        print(f"[PositionalEncoding] Input shape: {x.shape}")
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        x = self.dropout(x)
        print(f"[PositionalEncoding] Output shape: {x.shape}")
        return x


class DifferentialTransformerBlock(nn.Module):
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
        print(f"[DifferentialTransformerBlock] Input shape: {x.shape}")
        residual = x
        attended = self.attn(self.norm(x)) + residual
        residual_two = attended
        attended = self.attn(self.norm(residual_two)) + residual_two
        print(f"[DifferentialTransformerBlock] Output shape: {attended.shape}")
        return attended


class DifferentialTransformer(nn.Module):
    def __init__(self, dim: int = 512, heads: int = 8, dropout: float = 0.1, lambda_init: float = 0.05, depth: int = 6, num_tokens: int = 30000, seq_len: int = 350):
        super(DifferentialTransformer, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.lambda_init = lambda_init
        self.depth = depth
        self.num_tokens = num_tokens
        self.seq_len = seq_len

        self.layers = nn.ModuleList([DifferentialTransformerBlock(dim, heads, dropout, lambda_init) for _ in range(depth)])
        self.embed = nn.Embedding(num_embeddings=num_tokens, embedding_dim=dim)
        self.norm = SimpleRMSNorm(dim)
        self.pos_encoding = PositionalEncoding(dim, seq_len, dropout)
        self.output_head = OutputHead(dim, num_tokens)

    def forward(self, x):
        print(f"[DifferentialTransformer] Input shape: {x.shape}")
        x = self.norm(self.embed(x))
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        output = self.output_head(x)
        print(f"[DifferentialTransformer] Output shape: {output.shape}")
        return output


class OutputHead(nn.Module):
    def __init__(self, dim, num_tokens):
        super().__init__()
        self.linear = nn.Linear(dim, num_tokens)

    def forward(self, x):
        print(f"[OutputHead] Input shape: {x.shape}")
        result = self.linear(x)
        print(f"[OutputHead] Output shape: {result.shape}")
        return result


# Data Preparation and Loading
class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ])
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])
        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }


# Assuming the training and model instantiation logic follows
warnings.filterwarnings('ignore')
config = {
    'batch_size': 8,
    'num_epochs': 3,
    'lr': 1e-4,
    'seq_len': 350,
    'd_model': 512,
    'lang_src': 'en',
    'lang_tgt': 'it',
    'model_folder': 'weights',
    'model_basename': 'tmodel_',
    'preload': None,
    'tokenizer_file': 'tokenizer_{0}.json',
    'experiment_name': 'runs/tmodel'
}

# Train and Validate the Model
def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset and split
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Tokenizer loading and dataset preparation
    tokenizer_src = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_tgt']))

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=config['batch_size'])

    model = DifferentialTransformer(dim=config['d_model'], heads=8, dropout=0.1, lambda_init=0.05, depth=6, num_tokens=tokenizer_tgt.get_vocab_size(), seq_len=config['seq_len']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"))

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        train_iterator = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{config['num_epochs']}")

        for batch in train_iterator:
            optimizer.zero_grad()
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            labels = batch['label'].to(device)

            output = model(encoder_input)
            output = output.reshape(-1, output.shape[-1])
            labels = labels.reshape(-1)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iterator.set_postfix(loss=(train_loss / len(train_iterator)))

        # Validation
        model.eval()
        val_loss = 0.0
        val_iterator = tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}/{config['num_epochs']}")
        with torch.no_grad():
            for batch in val_iterator:
                encoder_input = batch['encoder_input'].to(device)
                decoder_input = batch['decoder_input'].to(device)
                labels = batch['label'].to(device)

                output = model(encoder_input)
                output = output.reshape(-1, output.shape[-1])
                labels = labels.reshape(-1)
                loss = loss_fn(output, labels)

                val_loss += loss.item()
                val_iterator.set_postfix(loss=(val_loss / len(val_iterator)))

train_model(config)
