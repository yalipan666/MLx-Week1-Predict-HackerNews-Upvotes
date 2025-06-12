import gensim
from gensim.models import Word2Vec
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train_word2vec():
    # Read the text8 file
    sentences = gensim.models.word2vec.LineSentence('text8')
    
    # Train the model with optimized parameters
    model = Word2Vec(sentences,
                    vector_size=100,    # dimension of word vectors
                    window=5,           # context window size
                    min_count=5,        # ignore words that appear less than this
                    workers=4,          # number of CPU cores to use
                    sg=0,              # use cbow (not skip-gram) model
                    epochs=3,          # reduced from 5 to 3 epochs
                    batch_words=10000)  # increased batch size for faster training
    
    # Save the model
    model.save('word2vec_model.bin')
    return model

if __name__ == "__main__":
    model = train_word2vec()
    print("Word2Vec model trained and saved as 'word2vec_model.bin'") 