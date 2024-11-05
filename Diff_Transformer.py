import torch
from diff_layer import *


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

        # Reshape attention mask to be compatible with transformer layers
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        for layer in self.layers:
            x = layer(x, mask=attention_mask)

        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x
