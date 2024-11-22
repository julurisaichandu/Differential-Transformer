import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from differential import DifferentialTransformer

class SmallNERDataset(Dataset):
    def __init__(self):
        self.data = [
            ("Barack Obama was born in Hawaii", ["B-PER", "I-PER", "O", "O", "O", "B-LOC"]),
            ("Apple Inc. is a technology company", ["B-ORG", "I-ORG", "O", "O", "O", "O"]),
            ("Elon Musk founded SpaceX", ["B-PER", "I-PER", "O", "B-ORG"]),
            ("Microsoft was founded by Bill Gates", ["B-ORG", "O", "O", "O", "B-PER", "I-PER"]),
            ("Jeff Bezos founded Amazon", ["B-PER", "I-PER", "O", "B-ORG"]),
            ("Google is a search engine", ["B-ORG", "O", "O", "O", "O"]),
            ("Facebook was created by Mark Zuckerberg", ["B-ORG", "O", "O", "O", "B-PER", "I-PER"]),
        ]
        self.word_to_idx = {word: idx for idx, word in enumerate(sorted(set(word for sentence, _ in self.data for word in sentence.split())))}
        self.tag_to_idx = {"B-PER": 0, "I-PER": 1, "B-LOC": 2, "B-ORG": 3, "I-ORG": 4, "O": 5}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.idx_to_tag = {idx: tag for tag, idx in self.tag_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, tags = self.data[idx]
        words = sentence.split()
        word_indices = [self.word_to_idx[word] for word in words]
        tag_indices = [self.tag_to_idx[tag] for tag in tags]
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

# Step 3: Prepare the model, loss function, and optimizer
dim = 64  # Model embedding dimension
heads = 8  # Number of attention heads
dropout = 0.1  # Dropout rate
lambda_init = 0.05  # Initial value for lambda
num_tokens = len(SmallNERDataset().word_to_idx)  # Number of unique words in the dataset
num_tags = len(SmallNERDataset().tag_to_idx)  # Number of entity tags

model = DifferentialTransformerNER(dim=dim, heads=heads, dropout=dropout, lambda_init=lambda_init, depth=2, num_tokens=num_tokens, num_tags=num_tags)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Split dataset into training and testing sets
dataset = SmallNERDataset()
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Training loop
model.train()
train_losses = []
train_accuracies = []
for epoch in range(10):  # Number of epochs
    total_loss = 0
    correct = 0
    total = 0
    for word_indices, tag_indices in train_loader:
        optimizer.zero_grad()
        output = model(word_indices)  # Output shape: (batch_size, seq_len, num_tags)
        output = output.view(-1, num_tags)  # Reshape output for loss calculation
        tag_indices = tag_indices.view(-1)  # Flatten tag indices
        loss = loss_function(output, tag_indices[:output.shape[0]])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate accuracy
        predictions = torch.argmax(output, dim=-1)
        correct += (predictions == tag_indices).sum().item()
        total += tag_indices.size(0)
    accuracy = correct / total
    train_losses.append(total_loss / len(train_loader))
    train_accuracies.append(accuracy)
    print(f"Epoch {epoch + 1}, Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.4f}")

# Evaluation on test set
model.eval()
with torch.no_grad():
    for word_indices, tag_indices in test_loader:
        output = model(word_indices)
        predictions = torch.argmax(output, dim=-1)
        words = [dataset.idx_to_word[idx.item()] for idx in word_indices[0]]
        predicted_tags = [dataset.idx_to_tag[idx.item()] for idx in predictions[0]]
        actual_tags = [dataset.idx_to_tag[idx.item()] for idx in tag_indices[0]]
        print(f"Test Input: {' '.join(words)}")
        print(f"Predicted Tags: {predicted_tags}")
        print(f"Actual Tags: {actual_tags}")
