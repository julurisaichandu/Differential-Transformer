import torch
import torch.nn as nn
import torch.nn.functional as F
import math




class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x):

        if x.size(-1) != self.dim:
            if x.size(-1) < self.dim:

                pad_size = self.dim - x.size(-1)
                x = F.pad(x, (0, pad_size))
            else:

                x = x[..., :self.dim]

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm * self.scale


class swiGLU(nn.Module):
    def __init__(self, d_model, exp_factor = 8/3):
        super().__init__()
        d_ff = int(d_model * exp_factor)
        self.w1 = nn.Linear(d_model, d_ff, bias = False)
        self.w2 = nn.Linear(d_model, d_ff, bias = False)
        self.w3 = nn.Linear(d_ff, d_model, bias = False)

    def forward(self, x):
        swish = F.silu(self.w1(x))
        linear = self.w2(x)
        out = swish * linear
        return self.w3(out)


class DifferentialAttention(nn.Module):
    def __init__(self, d_model, d_head, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads

        # Ensure output dimension matches d_model. ---VERY IMPORTANT---
        self.q_proj = nn.Linear(d_model, 2 * d_head * n_heads, bias=False)
        self.k_proj = nn.Linear(d_model, 2 * d_head * n_heads, bias=False)
        self.v_proj = nn.Linear(d_model, d_head * n_heads, bias=False)
        self.o_proj = nn.Linear(d_head * n_heads, d_model, bias=False)


        self.norm = RMSNorm(d_model)
        self.lambda_param = nn.Parameter(torch.ones(n_heads) * 0.8)
        self.dropout = nn.Dropout(dropout)

    def compute_lambda(self, batch_size, seq_len):
        # Properly indented compute_lambda method
        return self.lambda_param.view(1, self.n_heads, 1, 1).expand(batch_size, self.n_heads, seq_len, seq_len)

    def forward(self, x, context=None, mask=None):
        batch_size, seq_len, _ = x.shape

        if context is None:
            context = x

        # Project inputs
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, 2, self.d_head)
        k = self.k_proj(context).view(batch_size, context.size(1), self.n_heads, 2, self.d_head)
        v = self.v_proj(context).view(batch_size, context.size(1), self.n_heads, self.d_head)

        # Split queries and keys
        q1, q2 = q[..., 0, :], q[..., 1, :]
        k1, k2 = k[..., 0, :], k[..., 1, :]

        # Reshape for attention computation
        q1, q2 = q1.permute(0, 2, 1, 3), q2.permute(0, 2, 1, 3)
        k1, k2 = k1.permute(0, 2, 1, 3), k2.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.d_head)
        score_1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        score_2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        attn1 = F.softmax(score_1, dim=-1)
        attn2 = F.softmax(score_2, dim=-1)

        # Compute differential attention
        lambda_val = self.compute_lambda(batch_size, seq_len)
        attn_diff = attn1 - lambda_val * attn2

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.expand(batch_size, self.n_heads, seq_len, seq_len)
            attn_diff = attn_diff.masked_fill(mask == float('-inf'), float('-inf'))

        attn_weights = F.softmax(attn_diff, dim=-1)
        out = torch.matmul(attn_weights, v)

        # Reshape and normalize
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, self.n_heads * self.d_head)

        out = self.norm(out)
        out = out * (1 - self.lambda_param.mean())
        out = self.o_proj(out)

        return out


class DifferentialTransformerLayer(nn.Module):
    def __init__(self, d_model, d_head, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads

        self.attention = DifferentialAttention(d_model, d_head, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, x, context=None, mask=None):
        normed_x = self.norm(x)
        attn_output = self.attention(normed_x, context=context, mask=mask)
        attn_output = self.dropout(attn_output)
        x = x + attn_output
        return x

class DifferentialTransformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_classes=4,
                 d_model=3072,
                 n_layers=28,
                 d_head=128,
                 n_heads=12,
                 max_seq_len=4096,
                 dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            DifferentialTransformerLayer(d_model, d_head, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.head = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, x, attention_mask=None):
        b, t = x.size()
        positions = torch.arange(t, device=x.device).unsqueeze(0).expand(b, -1)

        token_embed = self.token_embedding(x)
        pos_embed = self.position_embedding(positions)
        x = token_embed + pos_embed

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        for layer in self.layers:
            x = layer(x, mask=attention_mask)

        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x




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
        #print(f"Embedding output shape: {x.shape}")
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
        #print(f"FeedForwardNetwork linear1 output shape: {x.shape}")
        x = self.swiGLU(x)
        #print(f"FeedForwardNetwork swiGLU output shape: {x.shape}")
        x = self.dropout(x)
        x = self.linear2(x)
        #print(f"FeedForwardNetwork linear2 output shape: {x.shape}")
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_head, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.prenorm1 = RMSNorm(d_model)
        self.attention = DifferentialTransformerLayer(d_model, d_head, n_heads, dropout)
        self.prenorm2 = RMSNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attention(self.prenorm1(x))
        x = x + self.ffn(self.prenorm2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_head, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.prenorm1 = RMSNorm(d_model)
        self.self_attention = DifferentialTransformerLayer(d_model, d_head, n_heads, dropout)
        self.prenorm2 = RMSNorm(d_model)
        self.cross_attention = DifferentialTransformerLayer(d_model, d_head, n_heads, dropout)
        self.prenorm3 = RMSNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

    def forward(self, x, encoder_out, target_mask=None):
        normed_x = self.prenorm1(x)
        x = x + self.self_attention(normed_x, mask=target_mask)

        normed_x = self.prenorm2(x)
        x = x + self.cross_attention.attention(normed_x, context=encoder_out)

        normed_x = self.prenorm3(x)
        x = x + self.ffn(normed_x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, d_head, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, d_head, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            #print(f"Encoder layer {i} input shape: {x.shape}")
            x = layer(x)
            #print(f"Encoder layer {i} output shape: {x.shape}")
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, d_head, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, d_head, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

    def forward(self, x, encoder_output, target_mask=None):
        for i, layer in enumerate(self.layers):
            #print(f"Decoder layer {i} input shape: {x.shape}")
            x = layer(x, encoder_output, target_mask=target_mask)
            #print(f"Decoder layer {i} output shape: {x.shape}")
        return x


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_head=64, d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(vocab_size, d_model, max_seq_len, dropout)
        self.encoder = Encoder(n_layers, d_model, d_head, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, d_head, n_heads, d_ff, dropout)
        # Add output projection layer
        self.output_projection = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        # Return mask of shape [1, sz, sz]
        return mask.unsqueeze(0)

    def forward(self, src_tokens, tgt_tokens):
        src_embeddings = self.embedding_layer(src_tokens)
        tgt_embeddings = self.embedding_layer(tgt_tokens)

        encoder_output = self.encoder(src_embeddings)
        tgt_seq_len = tgt_tokens.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt_tokens.device)

        decoder_output = self.decoder(tgt_embeddings, encoder_output, target_mask=tgt_mask)
        # Project to vocabulary size
        output = self.output_projection(decoder_output)
        return output