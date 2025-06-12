import torch
import numpy as np
from word_embeddings import load_word_vectors

def test_embeddings():
    # Load the word vectors
    word_vectors = load_word_vectors()
    
    # 1. Basic similarity test
    print("\n1. Basic Similarity Test:")
    test_words = ['computer', 'technology', 'science', 'art', 'music']
    for word in test_words:
        if word in word_vectors:
            # Calculate cosine similarity with all other words
            word_vec = word_vectors[word]
            similarities = {}
            for other_word, other_vec in word_vectors.items():
                if other_word != word:
                    # Convert to numpy for easier calculation
                    similarity = torch.nn.functional.cosine_similarity(
                        word_vec.unsqueeze(0),
                        other_vec.unsqueeze(0)
                    ).item()
                    similarities[other_word] = similarity
            
            # Get top 5 most similar words
            similar_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\nWords similar to '{word}':")
            for similar_word, similarity in similar_words:
                print(f"  {similar_word}: {similarity:.4f}")
        else:
            print(f"Word '{word}' not in vocabulary")
    
    # 2. Word analogies
    print("\n2. Word Analogies:")
    analogies = [
        ('king', 'man', 'woman'),  # king - man + woman = queen
        ('paris', 'france', 'germany'),  # paris - france + germany = berlin
        ('big', 'bigger', 'small')  # big - bigger + small = smaller
    ]
    
    for word1, word2, word3 in analogies:
        if all(word in word_vectors for word in [word1, word2, word3]):
            # Calculate analogy: word1 - word2 + word3
            vec1 = word_vectors[word1]
            vec2 = word_vectors[word2]
            vec3 = word_vectors[word3]
            result_vec = vec1 - vec2 + vec3
            
            # Find most similar word to result_vec
            similarities = {}
            for word, vec in word_vectors.items():
                if word not in [word1, word2, word3]:
                    similarity = torch.nn.functional.cosine_similarity(
                        result_vec.unsqueeze(0),
                        vec.unsqueeze(0)
                    ).item()
                    similarities[word] = similarity
            
            most_similar = max(similarities.items(), key=lambda x: x[1])
            print(f"\n{word1} - {word2} + {word3} = {most_similar[0]} (similarity: {most_similar[1]:.4f})")
        else:
            print(f"One of the words ({word1}, {word2}, {word3}) not in vocabulary")
    
    # 3. Clustering test
    print("\n3. Clustering Test:")
    tech_words = ['computer', 'software', 'hardware', 'internet', 'data']
    art_words = ['art', 'music', 'painting', 'dance', 'theater']
    
    def get_avg_vector(words):
        vectors = [word_vectors[word] for word in words if word in word_vectors]
        if vectors:
            return torch.stack(vectors).mean(dim=0)
        return None
    
    tech_vector = get_avg_vector(tech_words)
    art_vector = get_avg_vector(art_words)
    
    if tech_vector is not None and art_vector is not None:
        similarity = torch.nn.functional.cosine_similarity(
            tech_vector.unsqueeze(0),
            art_vector.unsqueeze(0)
        ).item()
        print(f"Similarity between tech and art clusters: {similarity:.4f}")
    
    # 4. Out-of-vocabulary test
    print("\n4. Out-of-vocabulary Test:")
    oov_words = ['supercalifragilisticexpialidocious', 'quantumcomputing', 'artificialintelligence']
    for word in oov_words:
        if word in word_vectors:
            print(f"Word '{word}' is in vocabulary (unexpected)")
        else:
            print(f"Word '{word}' is not in vocabulary (expected)")

if __name__ == "__main__":
    test_embeddings() 