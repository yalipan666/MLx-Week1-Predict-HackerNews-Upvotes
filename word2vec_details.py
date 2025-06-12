from gensim.models import Word2Vec
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def analyze_word2vec_implementation():
    # Load the model
    model = Word2Vec.load('word2vec_model.bin')
    
    # 1. Frequency Thresholding Analysis
    print("\n1. Frequency Thresholding Analysis:")
    print(f"Minimum word count threshold: {model.min_count}")
    print(f"Total vocabulary size: {len(model.wv)}")
    print(f"Total corpus size (total words): {model.corpus_count}")
    
    # 2. Subsampling Analysis
    print("\n2. Subsampling Analysis:")
    # Get the subsampling threshold (default is 1e-5 if not specified)
    threshold = getattr(model, 'sample', 1e-5)
    print(f"Subsampling threshold: {threshold}")
    
    # Calculate subsampling probabilities for some common words
    print("\nSubsampling probabilities for common words:")
    common_words = ['the', 'and', 'of', 'to', 'in', 'is', 'that', 'it', 'was', 'for']
    for word in common_words:
        if word in model.wv:
            try:
                # Get raw count and frequency
                raw_count = model.wv.get_vecattr(word, "count")
                # Debug information
                print(f"\nDebug for word '{word}':")
                print(f"  Raw count: {raw_count}")
                print(f"  Corpus count: {model.corpus_count}")
                print(f"  Raw count type: {type(raw_count)}")
                print(f"  Corpus count type: {type(model.corpus_count)}")
                
                # Calculate frequency
                freq = raw_count / model.corpus_count
                print(f"  Calculated frequency: {freq}")
                
                # Calculate subsampling probability
                prob = 1 - np.sqrt(threshold / freq)
                print(f"  Subsampling probability: {prob:.4f}")
            except (KeyError, AttributeError) as e:
                print(f"  {word}: Error - {str(e)}")
    
    # 3. Context Window Analysis
    print("\n3. Context Window Analysis:")
    print(f"Window size: {model.window}")
    print("Example of context window for word 'computer':")
    if 'computer' in model.wv:
        # Get most similar words
        similar_words = model.wv.most_similar('computer', topn=5)
        print("Most similar words:")
        for word, similarity in similar_words:
            print(f"  {word}: {similarity:.4f}")
    
    # 4. Training Parameters
    print("\n4. Training Parameters:")
    print(f"Vector size: {model.vector_size}")
    print(f"Training algorithm: {'Skip-gram' if model.sg else 'CBOW'}")
    print(f"Number of epochs: {model.epochs}")
    print(f"Number of workers: {model.workers}")
    
    # 5. Word Frequency Distribution
    print("\n5. Word Frequency Distribution:")
    # Get frequencies for all words
    frequencies = []
    raw_counts = []
    for word in model.wv.key_to_index:
        try:
            count = model.wv.get_vecattr(word, "count")
            freq = count / model.corpus_count
            frequencies.append(freq)
            raw_counts.append(count)
        except (KeyError, AttributeError):
            continue
    
    if frequencies:
        print("\nFrequency Statistics:")
        print(f"Min frequency: {min(frequencies):.6f}")
        print(f"Max frequency: {max(frequencies):.6f}")
        print(f"Mean frequency: {np.mean(frequencies):.6f}")
        
        # Plot frequency distribution
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Raw Counts
        plt.subplot(1, 2, 1)
        plt.hist(raw_counts, bins=50, log=True)
        plt.title('Word Count Distribution')
        plt.xlabel('Raw Count')
        plt.ylabel('Number of Words (log scale)')
        
        # Plot 2: Frequencies
        plt.subplot(1, 2, 2)
        plt.hist(frequencies, bins=50, log=True)
        plt.title('Word Frequency Distribution')
        plt.xlabel('Frequency')
        plt.ylabel('Number of Words (log scale)')
        
        plt.tight_layout()
        plt.savefig('word_distributions.png')
        print("Distribution plots saved as 'word_distributions.png'")
    else:
        print("Could not generate distribution plots - no frequency data available")

if __name__ == "__main__":
    analyze_word2vec_implementation() 