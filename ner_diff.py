import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from differential import DifferentialTransformer
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

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

def collate_fn(batch):
    word_indices, tag_indices = zip(*batch)
    word_indices_padded = pad_sequence(word_indices, batch_first=True, padding_value=0)
    tag_indices_padded = pad_sequence(tag_indices, batch_first=True, padding_value=-100)  # Use -100 for ignored index in loss
    return word_indices_padded, tag_indices_padded


dim = 64  # Model embedding dimension
heads = 8  # Number of attention heads
dropout = 0.2  # Dropout rate
lambda_init = 0.2  # Initial value for lambda

train_dataset = HuggingfaceNERDataset(split='train')
valid_dataset = HuggingfaceNERDataset(split='validation')
test_dataset = HuggingfaceNERDataset(split='test')

num_tokens = len(train_dataset.word_to_idx)  # Number of unique words in the dataset
num_tags = len(train_dataset.tag_to_idx)  # Number of entity tags

model = DifferentialTransformerNER(dim=dim, heads=heads, dropout=dropout, lambda_init=lambda_init, depth=2, num_tokens=num_tokens, num_tags=num_tags)

loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


model.train()
train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []

for epoch in range(20):

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


            predictions = torch.argmax(output, dim=-1)
            mask = tag_indices != -100  # Mask padding tokens
            correct += (predictions == tag_indices)[mask].sum().item()
            total += mask.sum().item()
    valid_accuracy = correct / total
    valid_losses.append(valid_loss / len(valid_loader))
    valid_accuracies.append(valid_accuracy)
    print(f"Epoch {epoch + 1}, Validation Loss: {valid_losses[-1]:.4f}, Validation Accuracy: {valid_accuracies[-1]:.4f}")
    model.train()

# Plot training and validation loss and accuracy
plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

#  Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(valid_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.tight_layout()
plt.show()


model.eval()
with torch.no_grad():
    for word_indices, tag_indices in test_loader:
        output = model(word_indices)
        predictions = torch.argmax(output, dim=-1)
        words = [train_dataset.idx_to_word[idx.item()] for idx in word_indices[0] if idx.item() in train_dataset.idx_to_word]
        predicted_tags = [train_dataset.idx_to_tag[idx.item()] for idx in predictions[0] if idx.item() in train_dataset.idx_to_tag]
        actual_tags = [train_dataset.idx_to_tag[idx.item()] for idx in tag_indices[0] if idx.item() in train_dataset.idx_to_tag]
        print(f"Test Input: {' '.join(words)}")
        print(f"Predicted Tags: {predicted_tags}")
        print(f"Actual Tags: {actual_tags}")
