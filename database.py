import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """
    Load data from the parquet file
    Returns the processed DataFrame
    """
    try:
        logging.info("Loading data from parquet file...")
        df = pd.read_parquet("hf://datasets/julien040/hacker-news-posts/story.parquet")
        
        # Process the data
        df['day_of_week'] = pd.to_datetime(df['time']).dt.day_name()
        df['day_of_week_num'] = pd.to_datetime(df['time']).dt.dayofweek
        df['hour_of_day'] = pd.to_datetime(df['time']).dt.hour
        df['log_10_score'] = np.log10(df['score'])
        
        logging.info(f"Loaded {len(df)} records")
        return df
    
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def prepare_data(df):
    """
    Prepare data for training
    Returns train_df, test_df, url_mapping, author_mapping
    """
    try:
        # Split into train and test sets
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Calculate URL and author frequencies
        url_freq = df['url'].value_counts()
        author_freq = df['author'].value_counts()
        
        # Create mappings for URLs and authors
        url_mapping = {url: idx for idx, url in enumerate(url_freq[url_freq >= 50].index)}
        author_mapping = {author: idx for idx, author in enumerate(author_freq[author_freq >= 5].index)}
        
        # Add unknown tokens
        url_mapping['UNKNOWN_URL'] = len(url_mapping)
        author_mapping['UNKNOWN_AUTHOR'] = len(author_mapping)
        
        logging.info(f"Created mappings for {len(url_mapping)} URLs and {len(author_mapping)} authors")
        return train_df, test_df, url_mapping, author_mapping
    
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        raise

if __name__ == "__main__":
    try:
        # Load and prepare data
        df = load_data()
        train_df, test_df, url_mapping, author_mapping = prepare_data(df)
        
        print(f"Training set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")
        print(f"Number of unique URLs: {len(url_mapping)}")
        print(f"Number of unique authors: {len(author_mapping)}")
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise 