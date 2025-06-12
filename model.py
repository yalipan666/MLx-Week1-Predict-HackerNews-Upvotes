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
    def __init__(self, titles, urls, authors, scores, word_vectors):
        self.titles = titles
        self.urls = urls
        self.authors = authors
        self.scores = scores
        self.word_vectors = word_vectors
        
    def __len__(self):
        return len(self.titles)
    
    def __getitem__(self, idx):
        title = self.titles[idx]
        url_idx = self.urls[idx]
        author_idx = self.authors[idx]
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
        
        return title_embedding, torch.tensor(url_idx, dtype=torch.long), torch.tensor(author_idx, dtype=torch.long), torch.tensor(score, dtype=torch.float)

class HackerNewsModel(nn.Module):
    def __init__(self, word_embedding_dim, num_urls, num_authors, url_embedding_dim=32, author_embedding_dim=32):
        super(HackerNewsModel, self).__init__()
        
        # URL and Author embeddings
        self.url_embedding = nn.Embedding(num_urls, url_embedding_dim)
        self.author_embedding = nn.Embedding(num_authors, author_embedding_dim)
        
        # Calculate total input dimension after concatenation
        total_input_dim = word_embedding_dim + url_embedding_dim + author_embedding_dim
        
        # Main network
        self.network = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, title_emb, url_idx, author_idx):
        # Get URL and author embeddings
        url_emb = self.url_embedding(url_idx)
        author_emb = self.author_embedding(author_idx)
        
        # Concatenate all embeddings
        combined = torch.cat([title_emb, url_emb, author_emb], dim=1)
        
        # Forward pass through network
        return self.network(combined)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for title_embs, url_indices, author_indices, targets in test_loader:
            title_embs = title_embs.to(device)
            url_indices = url_indices.to(device)
            author_indices = author_indices.to(device)
            outputs = model(title_embs, url_indices, author_indices)
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
        batch_size: Number of examples per batch (default: 512 for 16GB GPU), otherwise 32 to avoid memory error
        learning_rate: Learning rate for optimizer
        device: Device to train on (GPU/CPU)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load data
    data = load_data()
    X_train, X_test, train_urls, test_urls, train_authors, test_authors, y_train, y_test, url_mapping, author_mapping = prepare_data(data)
    
    # Load word vectors
    word_vectors = load_word_vectors(device)
    
    # Create datasets
    train_dataset = HackerNewsDataset(X_train, train_urls, train_authors, y_train, word_vectors)
    test_dataset = HackerNewsDataset(X_test, test_urls, test_authors, y_test, word_vectors)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    word_embedding_dim = next(iter(word_vectors.values())).size(0)
    model = HackerNewsModel(
        word_embedding_dim=word_embedding_dim,
        num_urls=len(url_mapping),
        num_authors=len(author_mapping)
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (title_embs, url_indices, author_indices, targets) in enumerate(train_loader):
            title_embs = title_embs.to(device)
            url_indices = url_indices.to(device)
            author_indices = author_indices.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(title_embs, url_indices, author_indices)
            loss = criterion(outputs.squeeze(), targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 600 == 0:  # Reduced logging frequency due to larger batch size
                logging.info(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for title_embs, url_indices, author_indices, targets in test_loader:
                title_embs = title_embs.to(device)
                url_indices = url_indices.to(device)
                author_indices = author_indices.to(device)
                targets = targets.to(device)
                
                outputs = model(title_embs, url_indices, author_indices)
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