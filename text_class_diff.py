import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from differential import DifferentialTransformer

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
DATASET_NAME = "ag_news"
dataset = load_dataset(DATASET_NAME)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Preprocess the dataset
def preprocess_data(examples):
    texts = examples['text']
    labels = examples['label']
    encoded_texts = tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    return {
        'input_ids': encoded_texts['input_ids'].squeeze(),
        'attention_mask': encoded_texts['attention_mask'].squeeze(),
        'labels': torch.tensor(labels, dtype=torch.long)
    }

# Apply preprocessing
dataset = dataset.map(preprocess_data, batched=True)

# Prepare DataLoader
def create_dataloader(split, batch_size=16):
    return DataLoader(dataset[split], batch_size=batch_size, shuffle=True)

train_loader = create_dataloader('train')
val_loader = create_dataloader('test')
test_loader = create_dataloader('test')

# Hyperparameters
num_tokens = tokenizer.vocab_size
dim = 512
heads = 8
depth = 6
dropout = 0.1
lambda_init = 0.8
num_epochs = 5
learning_rate = 1e-4

# Initialize model
model = DifferentialTransformer(
    dim=dim,
    heads=heads,
    dropout=dropout,
    lambda_init=lambda_init,
    depth=depth,
    num_tokens=num_tokens
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, num_tokens), labels.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, predicted = torch.max(outputs, -1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.numel()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    print(f'Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.4f}')

# Evaluation function
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, num_tokens), labels.view(-1))

            # Compute accuracy
            _, predicted = torch.max(outputs, -1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.numel()

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    print(f'Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy

# Training loop
for epoch in range(num_epochs):
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    train(model, train_loader, criterion, optimizer)
    evaluate(model, val_loader, criterion)

# Testing function
def pred_test(model, dataloader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids)
            _, predicted = torch.max(outputs, -1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.numel()

    accuracy = correct_predictions / total_predictions
    print(f'Test Accuracy: {accuracy:.4f}')

# Test the model
pred_test(model, test_loader)

# Save the model
torch.save(model.state_dict(), 'differential_transformer_model.pth')
