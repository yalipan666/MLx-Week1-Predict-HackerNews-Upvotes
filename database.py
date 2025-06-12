import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# set parameters
mini_count_url = 5
mini_count_author = 5

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
        
        logging.info(f"Loaded {len(df)} records")
        return df
    
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def prepare_data(df, mini_count_url=mini_count_url, mini_count_author=mini_count_author):
    """
    Prepare data for training
    Args:
        df: DataFrame containing the data
        mini_count_url: Minimum count for URLs to be included (default: 5)
        mini_count_author: Minimum count for authors to be included (default: 5)
    Returns:
        X_train, X_test, y_train, y_test, url_mapping, author_mapping
    """
    try:
        # Split into train and test sets
        train_df, test_df = train_test_split(df, test_size=0.05, random_state=42)
        
        # Calculate URL and author frequencies
        url_freq = df['url'].value_counts()
        author_freq = df['author'].value_counts()
        
        # Create mappings for URLs and authors
        url_mapping = {url: idx for idx, url in enumerate(url_freq[url_freq >= mini_count_url].index)}
        author_mapping = {author: idx for idx, author in enumerate(author_freq[author_freq >= mini_count_author].index)}
        
        # Add unknown tokens
        url_mapping['UNKNOWN_URL'] = len(url_mapping)
        author_mapping['UNKNOWN_AUTHOR'] = len(author_mapping)
        
        # Convert URLs and authors to indices
        train_urls = train_df['url'].map(lambda x: url_mapping.get(x, url_mapping['UNKNOWN_URL']))
        test_urls = test_df['url'].map(lambda x: url_mapping.get(x, url_mapping['UNKNOWN_URL']))
        train_authors = train_df['author'].map(lambda x: author_mapping.get(x, author_mapping['UNKNOWN_AUTHOR']))
        test_authors = test_df['author'].map(lambda x: author_mapping.get(x, author_mapping['UNKNOWN_AUTHOR']))
        
        logging.info(f"Created mappings for {len(url_mapping)} URLs and {len(author_mapping)} authors")
        
        # Return the split data with URL and author information
        return (
            train_df['title'].values, test_df['title'].values,
            train_urls.values, test_urls.values,
            train_authors.values, test_authors.values,
            train_df['score'].values, test_df['score'].values,
            url_mapping, author_mapping
        )
    
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        raise

if __name__ == "__main__":
    try:
        # Load and prepare data
        df = load_data()
        X_train, X_test, y_train, y_test, url_mapping, author_mapping = prepare_data(df)
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise 