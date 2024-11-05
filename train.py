import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from Diff_Transformer import DifferentialTransformer
from tqdm import tqdm


# Setting up hyperparameters

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 3
MAX_SEQ_LENGTH = 128
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

dataset = load_dataset("ag_news")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], padding = "max_length", truncation= True, max_length=MAX_SEQ_LENGTH)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=BATCH_SIZE)
test_dataloader = DataLoader(tokenized_dataset["test"], batch_size=BATCH_SIZE)

model = DifferentialTransformer(vocab_size = 30522)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr = LEARNING_RATE)

def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}") as t:
            for batch in t:
                input_ids = torch.tensor(batch["input_ids"]).to(DEVICE)
                attention_mask = torch.tensor(batch["attention_mask"]).to(DEVICE)
                labels = torch.tensor(batch["label"]).to(DEVICE)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")

def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = torch.tensor(batch["input_ids"]).to(DEVICE)
            attention_mask = torch.tensor(batch["attention_mask"]).to(DEVICE)
            labels = torch.tensor(batch["label"]).to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total}%")

if __name__ == "__main__":
    train()
    evaluate()
