import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
import numpy as np
from database import prepare_data_streaming, get_batch
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

class HackerNewsModel(nn.Module):
    def __init__(self, word_embedding_dim, num_urls, num_authors, url_embedding_dim=10, author_embedding_dim=10):
        super(HackerNewsModel, self).__init__()
        
        # URL and Author embeddings
        self.url_embedding = nn.Embedding(num_urls + 1, url_embedding_dim)  # +1 for unknown
        self.author_embedding = nn.Embedding(num_authors + 1, author_embedding_dim)  # +1 for unknown
        
        # Title embeddings will be pre-computed using Word2Vec
        
        # Final layers
        total_embedding_dim = word_embedding_dim + url_embedding_dim + author_embedding_dim
        self.fc1 = nn.Linear(total_embedding_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, title_emb, url_idx, author_idx):
        url_emb = self.url_embedding(url_idx)
        author_emb = self.author_embedding(author_idx)
        
        # Concatenate all embeddings
        combined = torch.cat([title_emb, url_emb, author_emb], dim=1)
        
        # Forward pass through fully connected layers
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def preprocess_title(title, word2vec_model):
    # Tokenize and clean the title
    words = re.findall(r'\b\w+\b', title.lower())
    
    # Get word vectors and average them
    word_vectors = []
    for word in words:
        try:
            word_vectors.append(word2vec_model.wv[word])
        except KeyError:
            continue
    
    if not word_vectors:
        return np.zeros(word2vec_model.vector_size)
    
    return np.mean(word_vectors, axis=0)

def train_model():
    # Get data generators and mappings
    train_gen, test_gen, url_mapping, author_mapping = prepare_data_streaming()
    
    # Load Word2Vec model
    word2vec_model = Word2Vec.load('word2vec_model.bin')
    
    # Initialize model
    model = HackerNewsModel(
        word_embedding_dim=word2vec_model.vector_size,
        num_urls=len(url_mapping),
        num_authors=len(author_mapping)
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    num_epochs = 500
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        # Process each chunk from the training generator
        for train_chunk in train_gen:
            # Get batch from the chunk
            batch_df = get_batch(train_chunk, batch_size=0.10)
            
            # Prepare batch data
            title_embs = torch.FloatTensor([
                preprocess_title(title, word2vec_model)
                for title in batch_df['title']
            ]).to(device)
            
            url_indices = torch.LongTensor([
                url_mapping.get(url, url_mapping['UNKNOWN_URL'])
                for url in batch_df['url']
            ]).to(device)
            
            author_indices = torch.LongTensor([
                author_mapping.get(author, author_mapping['UNKNOWN_AUTHOR'])
                for author in batch_df['author']
            ]).to(device)
            
            targets = torch.FloatTensor(batch_df['log_10_score'].values).view(-1, 1).to(device)
            
            # Forward pass
            outputs = model(title_embs, url_indices, author_indices)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Clean up
            del title_embs, url_indices, author_indices, targets, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Print progress
        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / num_batches
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
            
            # Find top 5 words similar to "computer"
            if (epoch + 1) % 500 == 0:
                try:
                    similar_words = word2vec_model.wv.most_similar('computer', topn=5)
                    logging.info("\nTop 5 words similar to 'computer':")
                    for word, similarity in similar_words:
                        logging.info(f"{word}: {similarity:.4f}")
                except KeyError:
                    logging.info("Word 'computer' not found in vocabulary")
    
    # Save the model
    torch.save(model.state_dict(), 'hackernews_model.pth')
    return model

if __name__ == "__main__":
    try:
        model = train_model()
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise 