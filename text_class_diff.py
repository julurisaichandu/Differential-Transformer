import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from differential import DifferentialTransformer
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load dataset
DATASET_NAME = "ag_news"
dataset = load_dataset(DATASET_NAME)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Preprocess the dataset
class TextClassificationDataset(Dataset):
    def __init__(self, split):
        self.data = dataset[split]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        encoded_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        return {
            'input_ids': encoded_text['input_ids'].squeeze(0),
            'attention_mask': encoded_text['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Custom collate function to pad sequences
def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Prepare DataLoader
def create_dataloader(split, batch_size=16):
    dataset_split = TextClassificationDataset(split)
    return DataLoader(dataset_split, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

train_loader = create_dataloader('train')
val_loader = create_dataloader('test')
test_loader = create_dataloader('test')
num_tokens = tokenizer.vocab_size
dim = 128
heads = 4
depth = 3
dropout = 0.1
lambda_init = 0.8
num_epochs = 5
learning_rate = 1e-3
batch_size = 32


model = DifferentialTransformer(
    dim=dim,
    heads=heads,
    dropout=dropout,
    lambda_init=lambda_init,
    depth=depth,
    num_tokens=num_tokens
).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0


    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids)
        outputs = outputs.mean(dim=1)
        outputs = nn.Linear(num_tokens, 4).to(device)(outputs)

        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        _, predicted = torch.max(outputs, -1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.numel()
        total_loss += loss.item()

        progress_bar.set_postfix({
            'loss': f'{total_loss / total_predictions:.4f}',
            'accuracy': f'{correct_predictions / total_predictions:.4f}'
        }, refresh=True)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)
            outputs = outputs.mean(dim=1)
            outputs = nn.Linear(num_tokens, 4).to(device)(outputs)

            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, -1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.numel()
            total_loss += loss.item()

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'accuracy': f'{correct_predictions / total_predictions:.4f}'
            })

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy


def pred_test(model, dataloader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(dataloader, desc="Testing", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)
            outputs = outputs.mean(dim=1)
            outputs = nn.Linear(num_tokens, 4).to(device)(outputs)

            _, predicted = torch.max(outputs, -1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.numel()

            progress_bar.set_postfix({
                'accuracy': f'{correct_predictions / total_predictions:.4f}'
            })

    accuracy = correct_predictions / total_predictions
    return accuracy

print("Starting training...")
epoch_progress = tqdm(range(num_epochs), desc="Epochs", position=0, leave=True)
for epoch in epoch_progress:
    # Training phase
    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation phase
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # Update epoch progress bar
    epoch_progress.set_postfix({
        'train_loss': f'{train_loss:.4f}',
        'train_acc': f'{train_acc:.4f}',
        'val_loss': f'{val_loss:.4f}',
        'val_acc': f'{val_acc:.4f}'
    }, refresh=True)

# Plotting
plt.figure(figsize=(12, 4))

# Plot losses
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Plot accuracies
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.close()

print("\nTesting model...")
test_accuracy = pred_test(model, test_loader)
print(f'Final Test Accuracy: {test_accuracy:.4f}')

torch.save(model.state_dict(), 'differential_transformer_model.pth')
print("\nModel saved successfully!")
print("Training metrics plot saved as 'training_metrics.png'")
