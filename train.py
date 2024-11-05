import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from Diff_Transformer import DifferentialTransformer
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 3
MAX_SEQ_LENGTH = 128
VOCAB_SIZE = 30522  # BERT-base vocab size
NUM_CLASSES = 4     # AG News has 4 classes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Model
model = DifferentialTransformer(
    vocab_size=VOCAB_SIZE,
    num_classes=NUM_CLASSES,
    d_model=3072,
    n_layers=28,
    d_head=128,
    n_heads=12,
    max_seq_len=MAX_SEQ_LENGTH,
    dropout=0.1
)
model.to(DEVICE)

dataset = load_dataset("ag_news")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: {
    "input_ids": torch.stack([torch.tensor(ex["input_ids"]) for ex in x]),
    "attention_mask": torch.stack([torch.tensor(ex["attention_mask"]) for ex in x]),
    "labels": torch.tensor([ex["label"] for ex in x])
})

test_dataloader = DataLoader(tokenized_dataset["test"], batch_size=BATCH_SIZE, collate_fn=lambda x: {
    "input_ids": torch.stack([torch.tensor(ex["input_ids"]) for ex in x]),
    "attention_mask": torch.stack([torch.tensor(ex["attention_mask"]) for ex in x]),
    "labels": torch.tensor([ex["label"] for ex in x])
})

# model = DifferentialTransformer(vocab_size=VOCAB_SIZE)
# model.to(DEVICE)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training Loop
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}") as t:
            for batch in t:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                t.set_postfix(loss=total_loss / (t.n + 1))

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")



def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            # Added Attention_mask reshaping here
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(-1, -1, attention_mask.size(-1), -1)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total}%")

# Start Training
if __name__ == "__main__":
    train()
    evaluate()
