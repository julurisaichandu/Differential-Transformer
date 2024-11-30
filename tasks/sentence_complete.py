import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from module.Multi_Head_Diff_Transformer import EncoderDecoderTransformer
from tqdm import tqdm


class SentenceCompletionDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.targets = []
        self.vocab_size = tokenizer.get_vocab_size()

        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue

            encoded = self.tokenizer.encode(text)
            tokens = encoded.ids

            tokens = [min(t, self.vocab_size - 1) for t in tokens]

            if len(tokens) > 8 and len(tokens) <= max_length:
                split_point = int(len(tokens) * 0.6)
                input_tokens = tokens[:split_point]
                target_tokens = tokens[split_point:]

                input_tokens = self._pad_sequence(input_tokens)
                target_tokens = self._pad_sequence(target_tokens)

                self.inputs.append(input_tokens)
                self.targets.append(target_tokens)

        print(f"Created dataset with {len(self.inputs)} samples")

    def _pad_sequence(self, seq):
        pad_token = self.tokenizer.token_to_id("[PAD]")
        if len(seq) < self.max_length:
            return seq + [pad_token] * (self.max_length - len(seq))
        return seq[:self.max_length]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])


def create_tokenizer(texts):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=30000,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        min_frequency=2
    )

    valid_texts = [text for text in texts if isinstance(text, str) and text.strip()]
    tokenizer.train_from_iterator(valid_texts, trainer)
    return tokenizer


def train_epoch(model, train_loader, optimizer, criterion, device, vocab_size):
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc='Training', leave=False)

    for batch_idx, (src, tgt) in enumerate(progress_bar):
        src = torch.clamp(src, 0, vocab_size - 1)
        tgt = torch.clamp(tgt, 0, vocab_size - 1)

        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()
        output = model(src, tgt)

        output = output.view(-1, vocab_size)
        tgt = tgt.view(-1)

        loss = criterion(output, tgt)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_description(f'Training (loss={loss.item():.4f})')

    return total_loss / len(train_loader)


def main():
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    print("Training tokenizer...")
    tokenizer = create_tokenizer(dataset['train']['text'])
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")

    print("Creating datasets...")
    train_dataset = SentenceCompletionDataset(dataset['train']['text'], tokenizer)
    val_dataset = SentenceCompletionDataset(dataset['validation']['text'], tokenizer)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Datasets are empty after processing!")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = EncoderDecoderTransformer(
        vocab_size=vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_head=64,
        d_ff=2048,
        max_seq_len=128,
        dropout=0.1
    ).to(device)

    def save_tokenizer(tokenizer, path):
        """Save the tokenizer to a file"""
        tokenizer.save(path)

    # In your main() function, after creating the tokenizer:
    save_tokenizer(tokenizer, "../tokenizer.json")
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    tokenizer = create_tokenizer(dataset['train']['text'])
    save_tokenizer(tokenizer, "../tokenizer.json")
    print("Starting training...")
    num_epochs = 10
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, vocab_size)
        print(f"Training loss: {train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in tqdm(val_loader, desc='Validation', leave=False):
                src = torch.clamp(src, 0, vocab_size - 1)
                tgt = torch.clamp(tgt, 0, vocab_size - 1)
                src, tgt = src.to(device), tgt.to(device)

                output = model(src, tgt)
                output = output.view(-1, output.size(-1))
                tgt = tgt.view(-1)
                val_loss += criterion(output, tgt).item()

        val_loss /= len(val_loader)
        print(f"Validation loss: {val_loss:.4f}")

        scheduler.step()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, f'checkpoint_epoch_{epoch}.pt')


if __name__ == "__main__":
    main()