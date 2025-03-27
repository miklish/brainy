import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from joblib.parallel import method
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST

class Expert(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  # Output is a single logit

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class GatingNetwork(nn.Module):
    def __init__(self, input_size=28*28, num_experts=10):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_experts)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

def bce_with_logits_loss(logits, targets):
    # Apply sigmoid to logits
    sigmoid_logits = torch.sigmoid(logits)

    # Compute binary cross-entropy loss
    loss = - (targets * torch.log(sigmoid_logits) + (1 - targets) * torch.log(1 - sigmoid_logits))

    # Return the mean loss
    return loss.mean()

def train_expert(expert, digit, train_dataset, epochs=5, batch_size=64, lr=0.001):
    # Filter dataset for the specific digit
    indices = [i for i, label in enumerate(train_dataset.targets) if label == digit]
    subset = Subset(train_dataset, indices)
    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    criterion = bce_with_logits_loss
    optimizer = optim.Adam(expert.parameters(), lr=lr)

    for epoch in range(epochs):
        for images, _ in train_loader:
            optimizer.zero_grad()
            outputs = expert(images).squeeze()
            labels = torch.ones(outputs.size())  # All labels are 1 (positive) for this digit
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def train_gating_network(gating_network, experts, train_dataset, epochs=5, batch_size=64, lr=0.001):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gating_network.parameters(), lr=lr)

    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            gating_outputs = gating_network(images)  # Shape: [batch_size, 10]
            expert_outputs = torch.stack([experts[digit](images).squeeze() for digit in range(10)], dim=1)  # Shape: [batch_size, 10]
            combined_outputs = gating_outputs * expert_outputs  # Element-wise multiplication
            predictions = combined_outputs.sum(dim=1)  # Sum over experts
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

experts = [Expert() for _ in range(10)]
gating_network = GatingNetwork()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def main():
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

    # Train the gating network
    train_gating_network(gating_network, experts, train_dataset)

    # Train all experts
    for digit in range(10):
        print(f'Training expert for digit {digit}')
        train_expert(experts[digit], digit, train_dataset)

if __name__ == "__main__":
    main()