from gensim.models import Word2Vec
import numpy as np

def analyze_model():
    # Load the trained model
    model = Word2Vec.load('word2vec_model.bin')
    
    # 1. Word Vectors Information
    print("\n1. Word Vectors Information:")
    print(f"Vector size (dimensions): {model.vector_size}")
    print(f"Total number of word vectors: {len(model.wv)}")
    print(f"Total size of word vectors in memory: {model.wv.vectors.nbytes / (1024*1024):.2f} MB")
    
    # 2. Vocabulary Information
    print("\n2. Vocabulary Information:")
    print(f"Total vocabulary size: {len(model.wv.key_to_index)}")
    print("\nSample of vocabulary words:")
    # Print first 10 words in vocabulary
    sample_words = list(model.wv.key_to_index.keys())[:10]
    for word in sample_words:
        print(f"  {word}")
    
    # 3. Model Parameters
    print("\n3. Model Parameters:")
    print(f"Window size: {model.window}")
    print(f"Minimum word count: {model.min_count}")
    print(f"Training algorithm: {'Skip-gram' if model.sg else 'CBOW'}")
    print(f"Number of epochs: {model.epochs}")
    print(f"Number of workers: {model.workers}")
    
    # 4. Sample Vector Analysis
    print("\n4. Sample Vector Analysis:")
    sample_word = 'computer'
    if sample_word in model.wv:
        vector = model.wv[sample_word]
        print(f"\nVector for '{sample_word}':")
        print(f"Shape: {vector.shape}")
        print(f"Mean: {np.mean(vector):.4f}")
        print(f"Std: {np.std(vector):.4f}")
        print(f"Min: {np.min(vector):.4f}")
        print(f"Max: {np.max(vector):.4f}")
    
    # 5. Memory Usage
    print("\n5. Memory Usage:")
    total_memory = model.wv.vectors.nbytes + sum(len(k) for k in model.wv.key_to_index.keys())
    print(f"Approximate total memory usage: {total_memory / (1024*1024):.2f} MB")

if __name__ == "__main__":
    analyze_model() 