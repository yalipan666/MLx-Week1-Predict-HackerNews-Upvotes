from gensim.models import Word2Vec
import numpy as np

def test_embeddings():
    # Load the trained model
    model = Word2Vec.load('word2vec_model.bin')
    
    # 1. Basic similarity test
    print("\n1. Basic Similarity Test:")
    test_words = ['computer', 'technology', 'science', 'art', 'music']
    for word in test_words:
        try:
            similar = model.wv.most_similar(word, topn=5)
            print(f"\nWords similar to '{word}':")
            for similar_word, similarity in similar:
                print(f"  {similar_word}: {similarity:.4f}")
        except KeyError:
            print(f"Word '{word}' not in vocabulary")
    
    # 2. Word analogies
    print("\n2. Word Analogies:")
    analogies = [
        ('king', 'man', 'woman'),  # king - man + woman = queen
        ('paris', 'france', 'germany'),  # paris - france + germany = berlin
        ('big', 'bigger', 'small')  # big - bigger + small = smaller
    ]
    
    for word1, word2, word3 in analogies:
        try:
            result = model.wv.most_similar(positive=[word1, word3], negative=[word2], topn=1)
            print(f"\n{word1} - {word2} + {word3} = {result[0][0]} (similarity: {result[0][1]:.4f})")
        except KeyError:
            print(f"One of the words ({word1}, {word2}, {word3}) not in vocabulary")
    
    # 3. Clustering test
    print("\n3. Clustering Test:")
    tech_words = ['computer', 'software', 'hardware', 'internet', 'data']
    art_words = ['art', 'music', 'painting', 'dance', 'theater']
    
    def get_avg_vector(words):
        vectors = [model.wv[word] for word in words if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else None
    
    tech_vector = get_avg_vector(tech_words)
    art_vector = get_avg_vector(art_words)
    
    if tech_vector is not None and art_vector is not None:
        similarity = np.dot(tech_vector, art_vector) / (np.linalg.norm(tech_vector) * np.linalg.norm(art_vector))
        print(f"Similarity between tech and art clusters: {similarity:.4f}")
    
    # 4. Out-of-vocabulary test
    print("\n4. Out-of-vocabulary Test:")
    oov_words = ['supercalifragilisticexpialidocious', 'quantumcomputing', 'artificialintelligence']
    for word in oov_words:
        try:
            model.wv[word]
            print(f"Word '{word}' is in vocabulary (unexpected)")
        except KeyError:
            print(f"Word '{word}' is not in vocabulary (expected)")

if __name__ == "__main__":
    test_embeddings() 