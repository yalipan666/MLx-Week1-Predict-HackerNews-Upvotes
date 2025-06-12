import pandas as pd 
import psycopg2 # postgreSQL database adapter
from sqlalchemy import create_engine  # for database connection
import numpy as np
from sklearn.model_selection import train_test_split
import gc  # For garbage collection
import time
from typing import Generator, Tuple, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_database_connection():
    """Create a database connection with error handling"""
    try:
        return psycopg2.connect(
            dbname="hd64m1ki",
            user="sy91dhb",
            password="g5t49ao",
            host="178.156.142.230",
            port="5432"
        )
    except psycopg2.Error as e:
        logging.error(f"Database connection error: {e}")
        raise

def stream_data_in_chunks(chunk_size: int = 10000) -> Generator[pd.DataFrame, None, None]:
    """
    Stream data in chunks without storing all chunks in memory
    Returns a generator that yields one chunk at a time
    """
    engine = None
    try:
        engine = create_engine('postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki')
        
        # Get total count
        count_query = """
        SELECT COUNT(*)
        FROM hacker_news.items
        WHERE
            type = 'story'
            AND title IS NOT NULL
            AND url IS NOT NULL
            AND score IS NOT NULL
            AND score >= 1
            AND (dead IS NULL OR dead = false)
        """
        total_count = pd.read_sql(count_query, engine).iloc[0, 0]
        logging.info(f"Total records to process: {total_count}")
        
        # Main query
        query = """
        SELECT
            to_char(time, 'Dy') AS day_of_week,
            to_char(time, 'D') AS day_of_week_num,
            to_char(time, 'HH') AS hour_of_day,
            LOG(score) AS log_10_score,
            score,
            title,
            url,
            "by" as author
        FROM hacker_news.items
        WHERE
            type = 'story'
            AND title IS NOT NULL
            AND url IS NOT NULL
            AND score IS NOT NULL
            AND score >= 1
            AND (dead IS NULL OR dead = false)
        """
        
        # Stream chunks
        for offset in range(0, total_count, chunk_size):
            try:
                chunk_query = f"{query} OFFSET {offset} LIMIT {chunk_size}"
                chunk = pd.read_sql(chunk_query, engine)
                logging.info(f"Loaded {min(offset + chunk_size, total_count)}/{total_count} rows")
                yield chunk
                
                # Force garbage collection after each chunk
                del chunk
                gc.collect()
                
            except Exception as e:
                logging.error(f"Error loading chunk at offset {offset}: {e}")
                # Retry logic
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        time.sleep(2 ** retry)  # Exponential backoff
                        chunk = pd.read_sql(chunk_query, engine)
                        logging.info(f"Successfully loaded chunk after retry {retry + 1}")
                        yield chunk
                        break
                    except Exception as retry_e:
                        if retry == max_retries - 1:
                            logging.error(f"Failed to load chunk after {max_retries} retries")
                            raise
                        continue
    
    except Exception as e:
        logging.error(f"Error in stream_data_in_chunks: {e}")
        raise
    finally:
        if engine:
            engine.dispose()

def process_chunks_to_mappings(chunk_size: int = 10000) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Process chunks to create URL and author mappings without storing all data in memory
    Returns URL and author mappings
    """
    url_counts = {}
    author_counts = {}
    
    try:
        for chunk in stream_data_in_chunks(chunk_size):
            # Update URL counts
            chunk_url_counts = chunk['url'].value_counts()
            for url, count in chunk_url_counts.items():
                url_counts[url] = url_counts.get(url, 0) + count
            
            # Update author counts
            chunk_author_counts = chunk['author'].value_counts()
            for author, count in chunk_author_counts.items():
                author_counts[author] = author_counts.get(author, 0) + count
            
            # Force garbage collection
            del chunk
            gc.collect()
        
        # Create mappings
        url_mapping = {url: idx for idx, url in enumerate(url for url, count in url_counts.items() if count >= 50)}
        author_mapping = {author: idx for idx, author in enumerate(author for author, count in author_counts.items() if count >= 5)}
        
        # Add unknown tokens
        url_mapping['UNKNOWN_URL'] = len(url_mapping)
        author_mapping['UNKNOWN_AUTHOR'] = len(author_mapping)
        
        return url_mapping, author_mapping
    
    except Exception as e:
        logging.error(f"Error in process_chunks_to_mappings: {e}")
        raise

def get_batch(df: pd.DataFrame, batch_size: float = 0.05) -> pd.DataFrame:
    """
    Get a random batch of data with memory optimization
    """
    try:
        batch_size = int(len(df) * batch_size)
        batch = df.sample(n=batch_size, random_state=np.random.randint(0, 1000))
        return batch
    except Exception as e:
        logging.error(f"Error in get_batch: {e}")
        raise

def prepare_data_streaming(chunk_size: int = 10000) -> Tuple[Generator[pd.DataFrame, None, None], 
                                                           Generator[pd.DataFrame, None, None], 
                                                           Dict[str, int], 
                                                           Dict[str, int]]:
    """
    Prepare data using streaming approach
    Returns generators for train and test data, and mappings
    """
    try:
        # Get mappings first
        url_mapping, author_mapping = process_chunks_to_mappings(chunk_size)
        
        # Create generators for train and test data
        def train_generator():
            for chunk in stream_data_in_chunks(chunk_size):
                # Split chunk into train/test
                train_chunk, _ = train_test_split(chunk, test_size=0.2, random_state=42)
                yield train_chunk
        
        def test_generator():
            for chunk in stream_data_in_chunks(chunk_size):
                # Split chunk into train/test
                _, test_chunk = train_test_split(chunk, test_size=0.2, random_state=42)
                yield test_chunk
        
        return train_generator(), test_generator(), url_mapping, author_mapping
    
    except Exception as e:
        logging.error(f"Error in prepare_data_streaming: {e}")
        raise

if __name__ == "__main__":
    try:
        # Example usage of streaming approach
        train_gen, test_gen, url_mapping, author_mapping = prepare_data_streaming()
        
        # Process first chunk of training data
        first_train_chunk = next(train_gen)
        print(f"First training chunk size: {len(first_train_chunk)}")
        
        # Process first chunk of test data
        first_test_chunk = next(test_gen)
        print(f"First test chunk size: {len(first_test_chunk)}")
        
        print(f"Number of unique URLs: {len(url_mapping)}")
        print(f"Number of unique authors: {len(author_mapping)}")
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise 