import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from module.RMSNorm import RMSNorm
from module.diff_layer import DifferentialTransformerLayer
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import optuna
import os


class DecoderOnlyModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_head, n_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer_layers = nn.ModuleList(
            [DifferentialTransformerLayer(d_model, d_head, n_heads, dropout) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        x = self.embedding_layer(x) + self.position_embedding(positions)
        x = self.dropout(x)
        for layer in self.transformer_layers:
            x = layer(x, mask=mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


def save_checkpoint(model, optimizer, epoch, loss, trial_number, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_trial_{trial_number}_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def objective(trial):
    # Hyperparameters to tune
    d_model = trial.suggest_categorical("d_model", [256, 512, 768])
    n_heads = trial.suggest_categorical("n_heads", [4, 8, 12])
    d_head = d_model // n_heads
    n_layers = trial.suggest_int("n_layers", 2, 6)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)


    model = DecoderOnlyModel(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_head=d_head,
        n_layers=n_layers,
        max_seq_len=128,
        dropout=dropout
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    model.train()
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    num_epochs = 20
    total_loss = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for batch in pbar:
            optimizer.zero_grad()


            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            if isinstance(input_ids, list):
                input_ids = torch.stack(
                    [x.clone().detach() if isinstance(x, torch.Tensor) else torch.tensor(x) for x in input_ids])
                attention_mask = torch.stack(
                    [x.clone().detach() if isinstance(x, torch.Tensor) else torch.tensor(x) for x in attention_mask])


            input_ids = input_ids.to(next(model.parameters()).device)
            attention_mask = attention_mask.to(input_ids.device)

            # Ensure dimensions match model's expected size
            if input_ids.size(-1) > 128:
                input_ids = input_ids[:, :128]
                attention_mask = attention_mask[:, :128]

            target_ids = input_ids[:, 1:]

            output = model(input_ids[:, :-1], mask=attention_mask[:, :-1])

            loss = criterion(
                output.reshape(-1, tokenizer.vocab_size),
                target_ids.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            current_loss = loss.item()
            epoch_loss += current_loss
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})

        avg_epoch_loss = epoch_loss / len(dataloader)
        total_loss += avg_epoch_loss

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'checkpoint_trial_{trial.number}_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)

    return total_loss / num_epochs


if __name__ == "__main__":

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token


    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=128,
            return_attention_mask=True
        )


    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    dataloader = DataLoader(tokenized_datasets, batch_size=32, shuffle=True)

    # Hyperparameter optimization
    try:

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10)

        # Print best parameters
        print("\nBest trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Save best model
        best_model = DecoderOnlyModel(
            vocab_size=tokenizer.vocab_size,
            d_model=trial.params["d_model"],
            n_heads=trial.params["n_heads"],
            d_head=trial.params["d_model"] // trial.params["n_heads"],
            n_layers=trial.params["n_layers"],
            max_seq_len=128,
            dropout=trial.params["dropout"]
        )


        best_checkpoint = torch.load(f'checkpoint_trial_{trial.number}_epoch_20.pt')
        best_model.load_state_dict(best_checkpoint['model_state_dict'])


        torch.save(best_model.state_dict(), 'best_model.pt')

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

    checkpoint_path = f"checkpoints/checkpoint_trial_{trial.number}_epoch_{study.best_trial.number}.pt"
    best_model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    best_model.eval()

    eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    eval_dataloader = DataLoader(tokenized_eval_dataset, batch_size=32)

    total_eval_loss = 0
    eval_criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(next(best_model.parameters()).device)
            target_ids = input_ids[:, 1:]
            output = best_model(input_ids[:, :-1])
            loss = eval_criterion(output.reshape(-1, tokenizer.vocab_size), target_ids.reshape(-1))
            total_eval_loss += loss.item()

    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    print(f"\nAverage Evaluation Loss: {avg_eval_loss:.4f}")