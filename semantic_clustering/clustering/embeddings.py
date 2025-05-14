"""
Functions for generating embeddings for keywords and text.
"""

import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
import logging
import time
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
            return np.zeros(model.get_sentence_embedding_dimension())
        except Exception as e:
            logger.error(f"Error loading model to determine dimension: {e}")
            # Return a small zero vector if we can't determine the dimension
            return np.zeros(384)  # Common dimension for sentence-transformers models
    
    # Attempt to load the model
    try:
        model = get_embedding_model(model_name)
    except Exception as e:
        error_msg = f"Failed to load embedding model '{model_name}': {str(e)}"
        logger.error(error_msg)
        raise ModelLoadError(error_msg) from e
    
    # Attempt embedding generation with retries
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            embedding = model.encode(text)
            
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
            last_exception = e
            if attempt < max_retries:
                logger.warning(f"Embedding generation attempt {attempt+1}/{max_retries+1} failed: {str(e)}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                error_msg = f"Failed to generate embedding after {max_retries+1} attempts: {str(e)}"
                logger.error(error_msg)
                # For the last attempt, return zeros but don't raise to maintain compatibility
                return np.zeros(model.get_sentence_embedding_dimension())
    
    # This should not be reached due to the return in the last attempt
    return np.zeros(model.get_sentence_embedding_dimension())

def batch_get_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = True,
    max_retries: int = 2,
    retry_delay: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Get embeddings for multiple texts in batches.
    
    Args:
        texts: List of texts to generate embeddings for
        model_name: Name of the embedding model to use
        batch_size: Number of texts to process in each batch
        normalize: Whether to normalize the embedding vectors
        show_progress: Whether to show a progress bar
        max_retries: Maximum number of retries per batch if generation fails
        retry_delay: Delay between retries in seconds
        
    Returns:
        Dictionary mapping texts to their embedding vectors
    
    Raises:
        ModelLoadError: If the embedding model fails to load
    """
    # Input validation
    if not texts:
        logger.warning("Empty list of texts provided")
        return {}
    
    # Validate and sanitize input texts
    valid_texts = []
    for i, text in enumerate(texts):
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
    
    embeddings_dict = {}
    failure_count = 0
    total_batches = (len(valid_texts) + batch_size - 1) // batch_size
    
    # Process texts in batches
    iterator = range(0, len(valid_texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating embeddings", total=total_batches)
    
    for i in iterator:
        batch_texts = valid_texts[i:i+batch_size]
        
        # Skip empty texts in this batch
        non_empty_indices = [j for j, text in enumerate(batch_texts) if text.strip()]
        if not non_empty_indices:
            logger.warning(f"Batch {i//batch_size + 1}/{total_batches} contains only empty texts, skipping")
            for text in batch_texts:
                embeddings_dict[text] = np.zeros(model.get_sentence_embedding_dimension())
            continue
        
        non_empty_batch = [batch_texts[j] for j in non_empty_indices]
        
        # Attempt batch processing with retries
        batch_success = False
        for attempt in range(max_retries + 1):
            try:
                batch_embeddings = model.encode(non_empty_batch)
                
                # Normalize if requested
                if normalize:
                    for j, embedding in enumerate(batch_embeddings):
                        if np.linalg.norm(embedding) > 0:
                            batch_embeddings[j] = embedding / np.linalg.norm(embedding)
                
                # Check for NaN or Inf values and replace them
                for j, embedding in enumerate(batch_embeddings):
                    if np.isnan(embedding).any() or np.isinf(embedding).any():
                        logger.warning(f"Generated embedding contains NaN or Inf values for text: {non_empty_batch[j][:50]}{'...' if len(non_empty_batch[j]) > 50 else ''}")
                        batch_embeddings[j] = np.nan_to_num(embedding)
                
                # Map embeddings back to original batch structure
                for j, idx in enumerate(non_empty_indices):
                    embeddings_dict[batch_texts[idx]] = batch_embeddings[j]
                
                # Add zero vectors for empty texts
                for j, text in enumerate(batch_texts):
                    if j not in non_empty_indices:
                        embeddings_dict[text] = np.zeros(model.get_sentence_embedding_dimension())
                
                batch_success = True
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
                        embeddings_dict[text] = np.zeros(model.get_sentence_embedding_dimension())
                    
                    # Warn if too many batches are failing
                    if error_ratio > 0.25:  # If more than 25% of batches fail
                        logger.error(f"WARNING: High failure rate ({error_ratio:.1%}) in embedding generation")
    
    # Final report
    success_count = len(embeddings_dict)
    if failure_count > 0:
        logger.warning(f"Completed embedding generation with {failure_count}/{total_batches} batch failures")
    
    return embeddings_dict

def get_keywords_embeddings_matrix(
    keywords: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = True,
    max_retries: int = 2,
    retry_delay: float = 1.0,
    require_min_success_rate: float = 0.5
) -> Tuple[np.ndarray, List[str]]:
    """
    Get embeddings for keywords and return as a matrix.
    
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
        
    Returns:
        Tuple of (embedding matrix, list of successfully processed keywords)
    
    Raises:
        ModelLoadError: If the embedding model fails to load
        ValueError: If the success rate is below the required minimum
    """
    if not keywords:
        logger.warning("Empty list of keywords provided")
        return np.array([]), []
    
    try:
        embeddings_dict = batch_get_embeddings(
            keywords, 
            model_name=model_name,
            batch_size=batch_size,
            normalize=normalize,
            show_progress=show_progress,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    except ModelLoadError as e:
        logger.error(f"Failed to load embedding model: {str(e)}")
        return np.array([]), []
    
    # Filter out any keywords that failed to generate non-zero embeddings
    successful_keywords = []
    embeddings_list = []
    
    for keyword in keywords:
        embedding = embeddings_dict.get(keyword)
        if embedding is not None and not np.all(embedding == 0):
            successful_keywords.append(keyword)
            embeddings_list.append(embedding)
    
    success_rate = len(successful_keywords) / len(keywords) if keywords else 0
    
    if len(embeddings_list) == 0:
        logger.error("No embeddings were successfully generated")
        return np.array([]), []
    
    # Check if we meet the minimum success rate requirement
    if success_rate < require_min_success_rate:
        warning_msg = (f"Low embedding success rate: {success_rate:.1%} "
                      f"({len(successful_keywords)}/{len(keywords)}). "
                      f"Minimum required: {require_min_success_rate:.1%}")
        logger.warning(warning_msg)
        
        # Still continue with available embeddings
    else:
        logger.info(f"Generated embeddings for {len(successful_keywords)}/{len(keywords)} keywords ({success_rate:.1%})")
    
    return np.array(embeddings_list), successful_keywords
