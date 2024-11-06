import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import numpy as np
from Multi_Head_Diff_Transformer import EncoderDecoderTransformer


class SentenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.targets = []

        for text in texts:
            # We skip empty strings
            if len(text.split()) < 4:
                continue

            encoded = self.tokenizer.encode(text)
            tokens = encoded.ids

            if len(tokens) > 4 and len(tokens) <= max_length:
                split_point = len(tokens) // 2
                input_tokens = tokens[: split_point]
                target_tokens = tokens[split_point :]

                input_tokens = self._pad_sequence(input_tokens)
                target_tokens = self._pad_sequence(target_tokens)


    def _pad_sequence(self, seq):
        if len(seq) < self.max_length:
            return seq + [self.tokenizer.token_to_id("[PAD]")] * (self.max_length - len(seq))

        return seq[:self.max_length]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

def train_tokenizer(texts):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])

    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer

def train_model(model, train_loader, val_loader, num_epochs, device, learning_rate=1e-4):
    criterion = nn.CrossEntropyLoss(ignore_index=0) # PAD token at index 0
    optimizer = torch.optim.AdamW(model.paramerters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)


    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (src, target) in enumerate(train_loader):
            src, target = src.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(src, target)

            output = output.view(-1, output.size(-1))
            target = target.view(-1)

            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.paramerters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item(): .4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss: .4f}")

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for src, target in val_loader:
                src, target = src.to(device) , target.to(device)
                output = model(src, target)
                output = output.view(-1, output.size(-1))
                target = target.view(-1)
                val_loss += criterion(output, target).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss: .4f}")





