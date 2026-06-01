import os
import numpy as np

def train_and_save():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    print("=== MNIST Model Training Script ===")
    
    # 1. Ensure output directory exists
    os.makedirs('assets', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # 2. Define transforms and download dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Mean and std of MNIST
    ])
    
    print("Downloading/Loading MNIST dataset...")
    try:
        train_dataset = datasets.MNIST(root='./data_mnist', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data_mnist', train=False, download=True, transform=transform)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Attempting to use custom dataset mirror or fallback...")
        # PyTorch will automatically try mirrors, but if it fails completely, we throw exception
        raise e
        
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 3. Define a simple 3-layer MLP
    # Input: 784 -> Hidden 1: 128 -> Hidden 2: 64 -> Output: 10
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
            
        def forward(self, x):
            x = x.view(-1, 784)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    # 4. Training Loop (approx 3-5 epochs for speed and high accuracy)
    epochs = 5
    print(f"Training MLP for {epochs} epochs on CPU...")
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if (batch_idx + 1) % 200 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {running_loss/200:.4f}")
                running_loss = 0.0
                
        acc = 100. * correct / total
        print(f"Epoch {epoch+1} complete. Training Accuracy: {acc:.2f}%")
        
    # 5. Evaluate Model
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
    print(f"Test Set Evaluation: Average Loss: {test_loss/len(test_loader):.4f}, Accuracy: {100. * correct / total:.2f}%")
    
    # 6. Extract weights and biases to export to NumPy
    print("Exporting model weights to NumPy format...")
    weights = {
        'W1': model.fc1.weight.detach().numpy(), # shape: (128, 784)
        'b1': model.fc1.bias.detach().numpy(),   # shape: (128,)
        'W2': model.fc2.weight.detach().numpy(), # shape: (64, 128)
        'b2': model.fc2.bias.detach().numpy(),   # shape: (64,)
        'W3': model.fc3.weight.detach().numpy(), # shape: (10, 64)
        'b3': model.fc3.bias.detach().numpy()    # shape: (10,)
    }
    
    np.savez_compressed('assets/weights.npz', **weights)
    print("Weights successfully saved to 'assets/weights.npz'!")
    print(f"W1 shape: {weights['W1'].shape}, b1 shape: {weights['b1'].shape}")
    print(f"W2 shape: {weights['W2'].shape}, b2 shape: {weights['b2'].shape}")
    print(f"W3 shape: {weights['W3'].shape}, b3 shape: {weights['b3'].shape}")
    print("=== Training Complete ===")

if __name__ == '__main__':
    train_and_save()
