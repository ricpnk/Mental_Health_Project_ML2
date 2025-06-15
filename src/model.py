import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

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
    
    # Initialize TensorBoard writer
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', f'training_{current_time}')
    writer = SummaryWriter(log_dir)
    
    # Log model architecture
    dummy_input = torch.randn(1, model.input_layer.in_features).to(device)
    writer.add_graph(model, dummy_input)
    
    best_accuracy = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
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
            
            # Log batch-level metrics
            if batch_idx % 10 == 0:  # Log every 10 batches
                writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + batch_idx)
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Evaluate on test set
        test_loss, test_accuracy, precision, recall, f1, roc_auc = evaluate(model, test_loader, loss_func, device)
        
        # Log epoch-level metrics
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        writer.add_scalar('Metrics/precision', precision, epoch)
        writer.add_scalar('Metrics/recall', recall, epoch)
        writer.add_scalar('Metrics/f1_score', f1, epoch)
        writer.add_scalar('Metrics/roc_auc', roc_auc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
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
    print(f"TensorBoard logs saved in: {log_dir}")
    
    writer.close()
    return model

def evaluate(model, test_loader, loss_func, device='cpu'):
    """
    Evaluate the model with comprehensive binary classification metrics
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    # Lists to store predictions and true labels for metrics calculation
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    # evaluate the model
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # forward pass
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            loss = loss_func(outputs, labels)
            total_loss += loss.item()
            
            # calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # store predictions and labels for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of positive class
    
    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    
    # Additional metrics
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    roc_auc = roc_auc_score(all_labels, all_probabilities)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Print detailed metrics
    print("\nDetailed Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    return avg_loss, accuracy, precision, recall, f1, roc_auc


def save_model(model, path):
    """
    Save the model to a file
    """
    torch.save(model.state_dict(), path)
    print(f"Best model saved to {path}")