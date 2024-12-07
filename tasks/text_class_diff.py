import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from differential import DifferentialTransformer
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("ag_news")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces=True)


class TextClassificationDataset(Dataset):
    def __init__(self, split):
        self.data = dataset[split]
        self.tokenizer = tokenizer
        self.max_length = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class TextClassificationTransformer(nn.Module):
    def __init__(self, num_classes, dim=128, heads=4, depth=3, num_tokens=None):
        super().__init__()
        self.transformer = DifferentialTransformer(
            dim=dim,
            heads=heads,
            depth=depth,
            num_tokens=num_tokens
        )
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        # Average pooling over sequence length
        x = x.mean(dim=1)
        return self.classifier(x)


def create_dataloaders(batch_size=32):
    train_dataset = TextClassificationDataset('train')
    test_dataset = TextClassificationDataset('test')

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size)
    )


num_classes = 4  # AG News has 4 classes
num_epochs = 5
batch_size = 32
learning_rate = 1e-3

model = TextClassificationTransformer(
    num_classes=num_classes,
    num_tokens=tokenizer.vocab_size
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_loader, test_loader = create_dataloaders(batch_size)


def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'accuracy': f'{100 * correct / total:.2f}%'
        })

    return total_loss / len(dataloader), correct / total


print("Starting training...")
train_losses = []
train_accuracies = []

for epoch in trange(num_epochs, desc="Epochs"):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    print(f'Epoch {epoch + 1}/{num_epochs}:')
    print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.close()

torch.save(model.state_dict(), 'text_classification_model.pth')
print("\nTraining completed and model saved!")