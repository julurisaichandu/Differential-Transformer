import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from differential import DifferentialTransformer
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# Load a simple NER dataset from Huggingface
dataset = load_dataset("conll2003")

# Preprocess dataset to be used with our model
class HuggingfaceNERDataset(Dataset):
    def __init__(self, split):
        self.data = dataset[split]
        self.word_to_idx = {}
        self.tag_to_idx = {}
        self.idx_to_word = {}
        self.idx_to_tag = {}

        # Build vocabulary and tag set
        for example in self.data:
            for word in example['tokens']:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = len(self.word_to_idx)
            for tag in example['ner_tags']:
                if tag not in self.tag_to_idx:
                    self.tag_to_idx[tag] = len(self.tag_to_idx)

        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        self.idx_to_tag = {v: k for k, v in self.tag_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        word_indices = [self.word_to_idx[word] for word in example['tokens']]
        tag_indices = [example['ner_tags'][i] for i in range(len(example['tokens']))]
        return torch.tensor(word_indices, dtype=torch.long), torch.tensor(tag_indices, dtype=torch.long)

class DifferentialTransformerNER(DifferentialTransformer):
    def __init__(self, dim, heads, dropout, lambda_init, depth, num_tokens, num_tags):
        super(DifferentialTransformerNER, self).__init__(dim=dim, heads=heads, dropout=dropout, lambda_init=lambda_init, depth=depth, num_tokens=num_tokens)
        # Override the output head to predict NER tags instead of the entire vocabulary
        self.output_head = torch.nn.Linear(dim, num_tags)

    def forward(self, x):
        x = self.norm(self.embed(x))
        for i, layer in enumerate(self.layers):
            x = layer(x)
        output = self.output_head(x)
        return output

# Custom collate function to pad sequences
def collate_fn(batch):
    word_indices, tag_indices = zip(*batch)
    word_indices_padded = pad_sequence(word_indices, batch_first=True, padding_value=0)
    tag_indices_padded = pad_sequence(tag_indices, batch_first=True, padding_value=-100)  # Use -100 for ignored index in loss
    return word_indices_padded, tag_indices_padded

# Step 3: Prepare the model, loss function, and optimizer
dim = 32  # Reduced Model embedding dimension
heads = 4  # Reduced Number of attention heads
dropout = 0.3  # Increased Dropout rate for regularization
lambda_init = 0.05  # Initial value for lambda

# Regularization additions
weight_decay = 1e-5  # L2 Regularization parameter

# Dataset preparation
differential_train_dataset = HuggingfaceNERDataset(split='train')
differential_valid_dataset = HuggingfaceNERDataset(split='validation')
differential_test_dataset = HuggingfaceNERDataset(split='test')

num_tokens = len(differential_train_dataset.word_to_idx)  # Number of unique words in the dataset
num_tags = len(differential_train_dataset.tag_to_idx)  # Number of entity tags

# Differential Transformer Model
model_diff = DifferentialTransformerNER(dim=dim, heads=heads, dropout=dropout, lambda_init=lambda_init, depth=2, num_tokens=num_tokens, num_tags=num_tags)

# Regular Transformer Model (Comparison)
class RegularTransformerNER(torch.nn.Module):
    def __init__(self, dim, heads, dropout, depth, num_tokens, num_tags):
        super(RegularTransformerNER, self).__init__()
        self.embed = torch.nn.Embedding(num_tokens, dim)
        self.layers = torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dropout=dropout) for _ in range(depth)])
        self.norm = torch.nn.LayerNorm(dim)
        self.output_head = torch.nn.Linear(dim, num_tags)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        output = self.output_head(x)
        return output

model_regular = RegularTransformerNER(dim=dim, heads=heads, dropout=dropout, depth=2, num_tokens=num_tokens, num_tags=num_tags)

loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100)
optimizer_diff = optim.Adam(model_diff.parameters(), lr=0.001, weight_decay=weight_decay)  # Added weight decay for L2 regularization
optimizer_regular = optim.Adam(model_regular.parameters(), lr=0.001, weight_decay=weight_decay)  # Added weight decay for L2 regularization

train_loader = DataLoader(differential_train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(differential_valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(differential_test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Training loop function
def train_model(model, optimizer, train_loader, valid_loader, num_epochs=5):
    model.train()
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    for epoch in range(num_epochs):
        # Training Phase
        total_loss = 0
        correct = 0
        total = 0
        for word_indices, tag_indices in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            optimizer.zero_grad()
            output = model(word_indices)  # Output shape: (batch_size, seq_len, num_tags)
            output = output.view(-1, num_tags)  # Reshape output for loss calculation
            tag_indices = tag_indices.view(-1)  # Flatten tag indices
            loss = loss_function(output, tag_indices)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(output, dim=-1)
            mask = tag_indices != -100  # Mask padding tokens
            correct += (predictions == tag_indices)[mask].sum().item()
            total += mask.sum().item()
        train_accuracy = correct / total
        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch + 1}, Training Loss: {train_losses[-1]:.4f}, Training Accuracy: {train_accuracies[-1]:.4f}")

        # Validation Phase
        model.eval()
        valid_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for word_indices, tag_indices in tqdm(valid_loader, desc=f"Validation Epoch {epoch + 1}"):
                output = model(word_indices)
                output = output.view(-1, num_tags)
                tag_indices = tag_indices.view(-1)
                loss = loss_function(output, tag_indices)
                valid_loss += loss.item()

                # Calculate accuracy
                predictions = torch.argmax(output, dim=-1)
                mask = tag_indices != -100  # Mask padding tokens
                correct += (predictions == tag_indices)[mask].sum().item()
                total += mask.sum().item()
        valid_accuracy = correct / total
        valid_losses.append(valid_loss / len(valid_loader))
        valid_accuracies.append(valid_accuracy)
        print(f"Epoch {epoch + 1}, Validation Loss: {valid_losses[-1]:.4f}, Validation Accuracy: {valid_accuracies[-1]:.4f}")
        model.train()

    return train_losses, train_accuracies, valid_losses, valid_accuracies

# Train Differential Transformer
print("Training Differential Transformer")
diff_train_losses, diff_train_accuracies, diff_valid_losses, diff_valid_accuracies = train_model(model_diff, optimizer_diff, train_loader, valid_loader)

# Train Regular Transformer
print("Training Regular Transformer")
regular_train_losses, regular_train_accuracies, regular_valid_losses, regular_valid_accuracies = train_model(model_regular, optimizer_regular, train_loader, valid_loader)

# Plot training and validation loss and accuracy
plt.figure(figsize=(12, 10))

# Plot Loss
plt.subplot(2, 2, 1)
plt.plot(diff_train_losses, label='Differential Transformer - Training Loss')
plt.plot(diff_valid_losses, label='Differential Transformer - Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Differential Transformer Loss over Epochs')

plt.subplot(2, 2, 2)
plt.plot(regular_train_losses, label='Regular Transformer - Training Loss')
plt.plot(regular_valid_losses, label='Regular Transformer - Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Regular Transformer Loss over Epochs')

# Plot Accuracy
plt.subplot(2, 2, 3)
plt.plot(diff_train_accuracies, label='Differential Transformer - Training Accuracy')
plt.plot(diff_valid_accuracies, label='Differential Transformer - Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Differential Transformer Accuracy over Epochs')

plt.subplot(2, 2, 4)
plt.plot(regular_train_accuracies, label='Regular Transformer - Training Accuracy')
plt.plot(regular_valid_accuracies, label='Regular Transformer - Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Regular Transformer Accuracy over Epochs')

plt.tight_layout()
plt.show()

model_diff.eval()
model_regular.eval()
with torch.no_grad():
    for word_indices, tag_indices in test_loader:
        output_diff = model_diff(word_indices)
        predictions_diff = torch.argmax(output_diff, dim=-1)
        output_regular = model_regular(word_indices)
        predictions_regular = torch.argmax(output_regular, dim=-1)

        words = [differential_train_dataset.idx_to_word[idx.item()] for idx in word_indices[0] if idx.item() in differential_train_dataset.idx_to_word]
        predicted_tags_diff = [differential_train_dataset.idx_to_tag[idx.item()] for idx in predictions_diff[0] if idx.item() in differential_train_dataset.idx_to_tag]
        predicted_tags_regular = [differential_train_dataset.idx_to_tag[idx.item()] for idx in predictions_regular[0] if idx.item() in differential_train_dataset.idx_to_tag]
        actual_tags = [differential_train_dataset.idx_to_tag[idx.item()] for idx in tag_indices[0] if idx.item() in differential_train_dataset.idx_to_tag]
        print(f"Test Input: {' '.join(words)}")
        print(f"Differential Transformer Predicted Tags: {predicted_tags_diff}")
        print(f"Regular Transformer Predicted Tags: {predicted_tags_regular}")
        print(f"Actual Tags: {actual_tags}")
