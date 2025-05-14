"""
Functions for generating embeddings for keywords and text with optimized memory usage.
"""

import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any, Iterator
import logging
import time
import gc
from tqdm import tqdm

from ..nlp.models import get_embedding_model

logger = logging.getLogger(__name__)

class EmbeddingError(Exception):
    """Base exception for embedding generation errors."""
    pass

class ModelLoadError(EmbeddingError):
    """Exception raised when the embedding model fails to load."""
    pass

class EmbeddingGenerationError(EmbeddingError):
    """Exception raised when embedding generation fails."""
    pass

def get_embeddings(
    text: str, 
    model_name: str = "all-MiniLM-L6-v2",
    normalize: bool = True,
    max_retries: int = 2,
    retry_delay: float = 1.0
) -> np.ndarray:
    """
    Get embeddings for a single text.
    
    Args:
        text: Text to generate embeddings for
        model_name: Name of the embedding model to use
        normalize: Whether to normalize the embedding vectors
        max_retries: Maximum number of retries if embedding generation fails
        retry_delay: Delay between retries in seconds
        
    Returns:
        Embedding vector as numpy array
        
    Raises:
        ModelLoadError: If the embedding model fails to load
        EmbeddingGenerationError: If embedding generation fails after retries
    """
    # Input validation
    if not isinstance(text, str):
        logger.warning(f"Expected string input, got {type(text)}. Converting to string.")
        text = str(text)
    
    if not text.strip():
        logger.warning("Empty text provided for embedding generation")
        # Return zero vector with correct dimensionality
        try:
            model = get_embedding_model(model_name)
            return np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)  # Use float32 to reduce memory
        except Exception as e:
            logger.error(f"Error loading model to determine dimension: {e}")
            # Return a small zero vector if we can't determine the dimension
            return np.zeros(384, dtype=np.float32)  # Common dimension for sentence-transformers models
    
    # Attempt to load the model
    try:
        model = get_embedding_model(model_name)
    except Exception as e:
        error_msg = f"Failed to load embedding model '{model_name}': {str(e)}"
        logger.error(error_msg)
        raise ModelLoadError(error_msg) from e
    
    # Attempt embedding generation with retries
    for attempt in range(max_retries + 1):
        try:
            # Force using float32 for better memory usage
            embedding = model.encode(text, convert_to_numpy=True).astype(np.float32)
            
            # Normalize if requested
            if normalize and np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
            
            # Check for NaN or Inf values
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                logger.warning(f"Generated embedding contains NaN or Inf values for text: {text[:50]}{'...' if len(text) > 50 else ''}")
                # Replace NaN and Inf with zeros
                embedding = np.nan_to_num(embedding)
                
            return embedding
            
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Embedding generation attempt {attempt+1}/{max_retries+1} failed: {str(e)}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                error_msg = f"Failed to generate embedding after {max_retries+1} attempts: {str(e)}"
                logger.error(error_msg)
                # For the last attempt, return zeros but don't raise to maintain compatibility
                return np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)
    
    # This should not be reached due to the return in the last attempt
    return np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)

def chunk_iterator(items: List[Any], chunk_size: int) -> Iterator[List[Any]]:
    """
    Split a list into chunks to avoid memory issues with large lists.
    
    Args:
        items: The list to split
        chunk_size: The size of each chunk
        
    Yields:
        Chunks of the original list
    """
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]

def batch_get_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = True,
    max_retries: int = 2,
    retry_delay: float = 1.0,
    low_memory: bool = False
) -> Dict[str, np.ndarray]:
    """
    Get embeddings for multiple texts in batches with optimized memory usage.
    
    Args:
        texts: List of texts to generate embeddings for
        model_name: Name of the embedding model to use
        batch_size: Number of texts to process in each batch
        normalize: Whether to normalize the embedding vectors
        show_progress: Whether to show a progress bar
        max_retries: Maximum number of retries per batch if generation fails
        retry_delay: Delay between retries in seconds
        low_memory: If True, uses additional memory-saving techniques
        
    Returns:
        Dictionary mapping texts to their embedding vectors
    
    Raises:
        ModelLoadError: If the embedding model fails to load
    """
    # Input validation
    if not texts:
        logger.warning("Empty list of texts provided")
        return {}
    
    # Optimize: process in chunks to avoid loading all texts at once
    max_chunk_size = 10000 if low_memory else len(texts)
    embeddings_dict = {}
    
    # Process texts in chunks to avoid memory issues with large datasets
    chunk_idx = 0
    for chunk in chunk_iterator(texts, max_chunk_size):
        chunk_idx += 1
        logger.info(f"Processing chunk {chunk_idx} with {len(chunk)} texts")
        
        # Validate and sanitize input texts
        valid_texts = []
        for i, text in enumerate(chunk):
            if not isinstance(text, str):
                logger.warning(f"Item {i}: Expected string, got {type(text)}. Converting to string.")
                valid_texts.append(str(text))
            else:
                valid_texts.append(text)
        
        # Attempt to load the model
        try:
            model = get_embedding_model(model_name)
        except Exception as e:
            error_msg = f"Failed to load embedding model '{model_name}': {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
        
        failure_count = 0
        total_batches = (len(valid_texts) + batch_size - 1) // batch_size
        
        # Process texts in batches
        iterator = range(0, len(valid_texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Generating embeddings for chunk {chunk_idx}", total=total_batches)
        
        for i in iterator:
            batch_texts = valid_texts[i:i+batch_size]
            
            # Skip empty texts in this batch
            non_empty_indices = [j for j, text in enumerate(batch_texts) if text.strip()]
            if not non_empty_indices:
                logger.warning(f"Batch {i//batch_size + 1}/{total_batches} contains only empty texts, skipping")
                for text in batch_texts:
                    embeddings_dict[text] = np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)
                continue
            
            non_empty_batch = [batch_texts[j] for j in non_empty_indices]
            
            # Attempt batch processing with retries
            for attempt in range(max_retries + 1):
                try:
                    # Force float32 to save memory
                    batch_embeddings = model.encode(
                        non_empty_batch, 
                        convert_to_numpy=True, 
                        show_progress=False
                    ).astype(np.float32)
                    
                    # Normalize if requested (in-place where possible)
                    if normalize:
                        for j in range(len(batch_embeddings)):
                            norm = np.linalg.norm(batch_embeddings[j])
                            if norm > 0:
                                batch_embeddings[j] /= norm
                    
                    # Check for NaN or Inf values and replace them (in-place)
                    has_nan_inf = np.isnan(batch_embeddings).any() or np.isinf(batch_embeddings).any()
                    if has_nan_inf:
                        for j, text in enumerate(non_empty_batch):
                            if np.isnan(batch_embeddings[j]).any() or np.isinf(batch_embeddings[j]).any():
                                logger.warning(f"Generated embedding contains NaN or Inf values for text: {text[:50]}{'...' if len(text) > 50 else ''}")
                                batch_embeddings[j] = np.nan_to_num(batch_embeddings[j])
                    
                    # Map embeddings back to original batch structure
                    for j, idx in enumerate(non_empty_indices):
                        embeddings_dict[batch_texts[idx]] = batch_embeddings[j]
                    
                    # Add zero vectors for empty texts
                    for j, text in enumerate(batch_texts):
                        if j not in non_empty_indices:
                            embeddings_dict[text] = np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)
                    
                    break
                    
                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(f"Batch {i//batch_size + 1}/{total_batches} attempt {attempt+1}/{max_retries+1} failed: {str(e)}. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        failure_count += 1
                        error_ratio = failure_count / total_batches
                        logger.error(f"Batch {i//batch_size + 1}/{total_batches} failed after {max_retries+1} attempts: {str(e)}")
                        
                        # Add zero vectors for failed batch
                        for text in batch_texts:
                            embeddings_dict[text] = np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)
                        
                        # Warn if too many batches are failing
                        if error_ratio > 0.25:  # If more than 25% of batches fail
                            logger.error(f"WARNING: High failure rate ({error_ratio:.1%}) in embedding generation")
            
            # Optimize: force garbage collection after processing each batch
            if low_memory:
                gc.collect()
        
        # Final report for this chunk
        if failure_count > 0:
            logger.warning(f"Completed chunk {chunk_idx} with {failure_count}/{total_batches} batch failures")
        
        # Optimize: force garbage collection between chunks
        gc.collect()
    
    return embeddings_dict

def get_keywords_embeddings_matrix(
    keywords: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = True,
    max_retries: int = 2,
    retry_delay: float = 1.0,
    require_min_success_rate: float = 0.5,
    low_memory: bool = False
) -> Tuple[np.ndarray, List[str]]:
    """
    Get embeddings for keywords and return as a matrix with optimized memory usage.
    
    Args:
        keywords: List of keywords to generate embeddings for
        model_name: Name of the embedding model to use
        batch_size: Number of keywords to process in each batch
        normalize: Whether to normalize the embedding vectors
        show_progress: Whether to show a progress bar
        max_retries: Maximum number of retries per batch if generation fails
        retry_delay: Delay between retries in seconds
        require_min_success_rate: Minimum percentage of keywords that must have
                                 successful embeddings (0.0 to 1.0)
        low_memory: If True, uses additional memory-saving techniques
        
    Returns:
        Tuple of (embedding matrix, list of successfully processed keywords)
    
    Raises:
        ModelLoadError: If the embedding model fails to load
        ValueError: If the success rate is below the required minimum
    """
    if not keywords:
        logger.warning("Empty list of keywords provided")
        return np.array([], dtype=np.float32), []
    
    try:
        embeddings_dict = batch_get_embeddings(
            keywords, 
            model_name=model_name,
            batch_size=batch_size,
            normalize=normalize,
            show_progress=show_progress,
            max_retries=max_retries,
            retry_delay=retry_delay,
            low_memory=low_memory
        )
    except ModelLoadError as e:
        logger.error(f"Failed to load embedding model: {str(e)}")
        return np.array([], dtype=np.float32), []
    
    # Memory-optimized approach for building embeddings matrix
    # 1. First pass to determine embedding dimension and count successful keywords
    first_embedding = None
    successful_count = 0
    
    for keyword in keywords:
        embedding = embeddings_dict.get(keyword)
        if embedding is not None and not np.all(embedding == 0):
            if first_embedding is None:
                first_embedding = embedding
            successful_count += 1
    
    if successful_count == 0 or first_embedding is None:
        logger.error("No embeddings were successfully generated")
        return np.array([], dtype=np.float32), []
    
    # 2. Pre-allocate arrays for better memory efficiency
    embedding_dim = first_embedding.shape[0]
    embeddings_matrix = np.zeros((successful_count, embedding_dim), dtype=np.float32)
    successful_keywords = [None] * successful_count
    
    # 3. Fill the arrays in a single pass
    idx = 0
    for keyword in keywords:
        embedding = embeddings_dict.get(keyword)
        if embedding is not None and not np.all(embedding == 0):
            embeddings_matrix[idx] = embedding
            successful_keywords[idx] = keyword
            idx += 1
    
    # Clear the dictionary to free memory
    if low_memory:
        embeddings_dict.clear()
        gc.collect()
    
    # Calculate success rate
    success_rate = successful_count / len(keywords) if keywords else 0
    
    # Check if we meet the minimum success rate requirement
    if success_rate < require_min_success_rate:
        warning_msg = (f"Low embedding success rate: {success_rate:.1%} "
                      f"({successful_count}/{len(keywords)}). "
                      f"Minimum required: {require_min_success_rate:.1%}")
        logger.warning(warning_msg)
    else:
        logger.info(f"Generated embeddings for {successful_count}/{len(keywords)} keywords ({success_rate:.1%})")
    
    return embeddings_matrix, successful_keywords
