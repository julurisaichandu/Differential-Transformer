import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from RMSNorm import RMSNorm
from diff_layer import DifferentialTransformerLayer
from swiGLU import swiGLU
import optuna

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


if __name__ == "__main__":
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = AutoTokenizer.from_pretrained("gpt_2")

    def tokenize_function(examples):
        return tokenizer(examples["text"], return_tensors="pt", truncation=True, padding = 'max_length', max_length=128)


    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    dataloader = DataLoader(tokenized_datasets, batch_size=32, shuffle=True)

    def objective(trial):

        d_model = trial.suggest_categorical("d_model", [256, 512, 768])
        n_heads = trial.suggest_categorical("n_heads", [4, 8, 12])
        d_head = d_model // n_heads
        n_layers = trial.suggest_int("n_layers", 2, 6)
        dropout = trial.suggest_float("dropout",0.1 , 0.5)

        model = DecoderOnlyModel(vocab_size=len(tokenizer), d_model=d_model, n_heads = n_heads, d_head=d_head, n_layers=n_layers, max_seq_len=128, dropout=dropout)
        model.train()

        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
        criterion = nn.CrossEntropyLoss()

        num_epochs = 3
        total_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].squeeze(1)
                target_ids = batch["input_ids"].squeeze(1)
                output = model(input_ids)
                loss = criterion(output.view(-1,len(tokenizer)), target_ids.view(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            total_loss += epoch_loss / len(dataloader)
        return total_loss / num_epochs

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    best_hyparams = study.best_params
    print("Best Hyperparameters:", best_hyparams)


    model = DecoderOnlyModel(vocab_size = len(tokenizer), d_model=best_hyparams["d_model"], n_heads=best_hyparams["n_heads"], d_head=best_hyparams["d_model"] // best_hyparams["n_heads"], n_layers=best_hyparams["n_layers"], max_seq_len=128, dropout=best_hyparams["dropout"])
    model.eval()



