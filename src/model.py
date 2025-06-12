import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader

class MLPClassifier(nn.Module):
    """
    Multi-layer Perceptron Classifier with variable number of hidden layers
    """
    def __init__(self, input_dim, hidden_dim, dropout, output_dim, num_hidden_layers):
        """
        Initialize the model
        """
        super().__init__()
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Create hidden layers dynamically
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the model
        """
        # Input layer
        x = self.input_layer(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Hidden layers with residual connections
        for hidden_layer, batch_norm in zip(self.hidden_layers, self.batch_norms):
            identity = x
            x = hidden_layer(x)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + identity  # Residual connection
        
        # Output layer
        x = self.output_layer(x)
        return x
    


def train(model, train_loader, test_loader,  save_path, epochs=10, learning_rate=0.001, device='cpu'):
    """
    Train the model
    """
    model.train()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Increased weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    loss_func = nn.CrossEntropyLoss()
    
    best_accuracy = 0
    for epoch in range(epochs):
        model.train()
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
        
        # Evaluate and update learning rate
        accuracy = evaluate(model, test_loader, device)
        scheduler.step(accuracy)
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, save_path)
    
    return model, best_accuracy
    


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
    print(f"Best model saved to {path}")