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
    


def train(model, train_loader, test_loader, save_path, epochs=10, learning_rate=0.001, device='cpu'):
    """
    Train the model
    """
    model.train()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    loss_func = nn.CrossEntropyLoss()
    
    best_accuracy = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Evaluate on test set
        test_loss, test_accuracy = evaluate(model, test_loader, loss_func, device)
        
        scheduler.step(test_accuracy)
        
        print("=" * 50)
        print(f"Epoch: {epoch+1}")
        print(f"Train Loss: {avg_train_loss:10.4f} | Train Accuracy: {train_accuracy:9.2f}%") 
        print(f"Test Loss: {test_loss:11.4f} | Test Accuracy: {test_accuracy:10.2f}%")
        
        # Save best model
        if test_accuracy > best_accuracy:
            print("-" * 50)
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            save_model(model, save_path)
        print("=" * 50)
    
    print("-" * 50)
    print(f"\nTraining completed!")
    print(f"Best model achieved {best_accuracy:.2f}% accuracy at epoch {best_epoch}")
    
    return model

def evaluate(model, test_loader, loss_func, device='cpu'):
    """
    Evaluate the model
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    # evaluate the model
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # forward pass
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            total_loss += loss.item()
            
            # calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return avg_loss, accuracy


def save_model(model, path):
    """
    Save the model to a file
    """
    torch.save(model.state_dict(), path)
    print(f"Best model saved to {path}")