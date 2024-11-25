import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from differential import DifferentialTransformer
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange

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
dim = 128           # Reduced from 512
heads = 4           # Reduced from 8
depth = 3           # Reduced from 6
dropout = 0.1
lambda_init = 0.8
num_epochs = 5
learning_rate = 1e-3
batch_size = 32     # Increased batch size for faster training


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

        outputs = model(input_ids)
        batch_size = outputs.size(0)
        outputs = outputs.mean(dim=1)
        outputs = nn.Linear(num_tokens, 4).to(device)(outputs)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
for epoch in trange(num_epochs, desc="Epochs"):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    print(f'Epoch {epoch + 1}/{num_epochs}:')
    print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}')

    val_loss, val_acc = evaluate(model, val_loader, criterion)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
    print('-' * 50)

print("\nTesting model...")
test_accuracy = pred_test(model, test_loader)
print(f'Final Test Accuracy: {test_accuracy:.4f}')

torch.save(model.state_dict(), 'differential_transformer_model.pth')
print("\nModel saved successfully!")
