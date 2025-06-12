import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from word_embeddings import load_word_vectors
from database import load_data, prepare_data

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class HackerNewsDataset(Dataset):
    def __init__(self, titles, scores, word_vectors):
        self.titles = titles
        self.scores = scores
        self.word_vectors = word_vectors
        
    def __len__(self):
        return len(self.titles)
    
    def __getitem__(self, idx):
        title = self.titles[idx]
        score = self.scores[idx]
        
        # Convert title to embedding
        words = title.lower().split()
        word_embeddings = []
        for word in words:
            if word in self.word_vectors:
                word_embeddings.append(self.word_vectors[word])
        
        if word_embeddings:
            # Stack embeddings and take mean
            title_embedding = torch.stack(word_embeddings).mean(dim=0)
        else:
            # If no valid words, use zero vector
            title_embedding = torch.zeros_like(next(iter(self.word_vectors.values())))
        
        return title_embedding, torch.tensor(score, dtype=torch.float)

class HackerNewsModel(nn.Module):
    def __init__(self, input_size):
        super(HackerNewsModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_predictions.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # Calculate percentage of predictions within different error ranges
    error_ranges = {
        '±10': np.mean(np.abs(predictions - targets) <= 10) * 100,
        '±20': np.mean(np.abs(predictions - targets) <= 20) * 100,
        '±50': np.mean(np.abs(predictions - targets) <= 50) * 100,
        '±100': np.mean(np.abs(predictions - targets) <= 100) * 100
    }
    
    # Get some example predictions
    indices = np.random.choice(len(predictions), min(5, len(predictions)), replace=False)
    examples = [(predictions[i], targets[i]) for i in indices]
    
    # Print evaluation metrics
    logging.info("\nModel Evaluation Metrics:")
    logging.info(f"Root Mean Square Error (RMSE): {rmse:.2f} points")
    logging.info("\nPercentage of predictions within error ranges:")
    for range_name, percentage in error_ranges.items():
        logging.info(f"Within {range_name} points: {percentage:.2f}%")
    
    logging.info("\nExample Predictions (Predicted vs Actual):")
    for pred, actual in examples:
        logging.info(f"Predicted: {pred:.0f}, Actual: {actual:.0f}, Difference: {abs(pred - actual):.0f} points")
    
    return rmse

def train_model(epochs=10, batch_size=512, learning_rate=0.001, device=None):
    """
    Train the Hacker News model
    Args:
        epochs: Number of training epochs
        batch_size: Number of examples per batch (default: 512 for 16GB GPU), otherwise 32 to aviod memory error
        learning_rate: Learning rate for optimizer
        device: Device to train on (GPU/CPU)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load data
    data = load_data()
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # Load word vectors
    word_vectors = load_word_vectors(device)
    
    # Create datasets
    train_dataset = HackerNewsDataset(X_train, y_train, word_vectors)
    test_dataset = HackerNewsDataset(X_test, y_test, word_vectors)
    
    # Create data loaders with larger batch size for GPU efficiency
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    input_size = next(iter(word_vectors.values())).size(0)
    model = HackerNewsModel(input_size).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:  # Reduced logging frequency due to larger batch size
                logging.info(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs.squeeze(), targets).item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        logging.info(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # Final evaluation
    logging.info("\nFinal Model Evaluation:")
    rmse = evaluate_model(model, test_loader, device)
    
    # Save the model
    torch.save(model.state_dict(), 'hackernews_model.pth')
    logging.info("Model saved as 'hackernews_model.pth'")
    
    return model

if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Train the model with larger batch size for GPU efficiency
    model = train_model(epochs=10, batch_size=512, learning_rate=0.001, device=device) 