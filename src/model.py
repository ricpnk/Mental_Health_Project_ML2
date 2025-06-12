import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader

class MLPClassifier(nn.Module):
    """
    Multi-layer Perceptron Classifier
    """
    def __init__(self, input_dim, hidden_dim, dropout, output_dim):
        """
        Initialize the model
        """
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        """
        Forward pass of the model
        """
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        x = F.relu(self.hidden_layer(x))
        x = self.dropout(x)
        x = F.relu(self.hidden_layer(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
    


def train(model, train_loader, test_loader, epochs=10, learning_rate=0.001, device='cpu'):
    """
    Train the model
    """
    model.train()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        accuracy = evaluate(model, test_loader, device)
    return model, accuracy
    


def evaluate(model, test_loader, device='cpu'):
    """
    Evaluate the model
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')
    return accuracy


def save_model(model, path):
    """
    Save the model to a file
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")