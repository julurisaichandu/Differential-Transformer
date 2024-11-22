import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from differential import DifferentialTransformer

class SmallNERDataset(Dataset):
    def __init__(self):
        self.data = [
            ("Barack Obama was born in Hawaii", ["B-PER", "I-PER", "O", "O", "O", "B-LOC"]),
            ("Apple Inc. is a technology company", ["B-ORG", "I-ORG", "O", "O", "O", "O"]),
            ("Elon Musk founded SpaceX", ["B-PER", "I-PER", "O", "B-ORG"]),
        ]
        self.word_to_idx = {word: idx for idx, word in enumerate(sorted(set(word for sentence, _ in self.data for word in sentence.split())))}
        self.tag_to_idx = {"B-PER": 0, "I-PER": 1, "B-LOC": 2, "B-ORG": 3, "I-ORG": 4, "O": 5}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, tags = self.data[idx]
        words = sentence.split()
        word_indices = [self.word_to_idx[word] for word in words]
        tag_indices = [self.tag_to_idx[tag] for tag in tags]
        return torch.tensor(word_indices, dtype=torch.long), torch.tensor(tag_indices, dtype=torch.long)

dim = 64  # Model embedding dimension
heads = 8  # Number of attention heads
dropout = 0.1  # Dropout rate
lambda_init = 0.05  # Initial value for lambda
num_tokens = len(SmallNERDataset().word_to_idx)  # Number of unique words in the dataset
num_tags = len(SmallNERDataset().tag_to_idx)  # Number of entity tags

model = DifferentialTransformer(dim=dim, heads=heads, dropout=dropout, lambda_init=lambda_init, depth=2, num_tokens=num_tokens)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(SmallNERDataset(), batch_size=1, shuffle=True)

model.train()

for epoch in range(5):
    total_loss = 0
    for word_indices, tag_indices in train_loader:
        optimizer.zero_grad()
        output = model(word_indices)  # Output shape: (batch_size, seq_len, vocab_size)
        output = output.view(-1, output.shape[-1])  # Reshape output for loss calculation
        tag_indices = tag_indices.view(-1)  # Flatten tag indices
        loss = loss_function(output, tag_indices[:output.shape[0]])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

model.eval()
with torch.no_grad():
    for word_indices, tag_indices in train_loader:
        output = model(word_indices)
        predictions = torch.argmax(output, dim=-1)
        print(f"Input: {word_indices}")
        print(f"Predicted Tags: {predictions}")
        print(f"Actual Tags: {tag_indices}")
