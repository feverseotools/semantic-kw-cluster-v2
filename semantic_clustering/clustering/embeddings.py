"""
Functions for generating embeddings for keywords and text.
"""

import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import logging
from tqdm import tqdm

from ..nlp.models import get_embedding_model

logger = logging.getLogger(__name__)

def get_embeddings(
    text: str, 
    model_name: str = "all-MiniLM-L6-v2",
    normalize: bool = True
) -> np.ndarray:
    """
    Get embeddings for a single text.
    
    Args:
        text: Text to generate embeddings for
        model_name: Name of the embedding model to use
        normalize: Whether to normalize the embedding vectors
        
    Returns:
        Embedding vector as numpy array
    """
    model = get_embedding_model(model_name)
    
    # Generate embedding
    try:
        embedding = model.encode(text)
        
        # Normalize if requested
        if normalize and np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
            
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding for text: {e}")
        # Return a zero vector with the same dimensionality as the model output
        return np.zeros(model.get_sentence_embedding_dimension())

def batch_get_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """
    Get embeddings for multiple texts in batches.
    
    Args:
        texts: List of texts to generate embeddings for
        model_name: Name of the embedding model to use
        batch_size: Number of texts to process in each batch
        normalize: Whether to normalize the embedding vectors
        show_progress: Whether to show a progress bar
        
    Returns:
        Dictionary mapping texts to their embedding vectors
    """
    model = get_embedding_model(model_name)
    embeddings_dict = {}
    
    # Process texts in batches
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating embeddings")
        
    for i in iterator:
        batch_texts = texts[i:i+batch_size]
        try:
            batch_embeddings = model.encode(batch_texts)
            
            # Normalize if requested
            if normalize:
                for j, embedding in enumerate(batch_embeddings):
                    if np.linalg.norm(embedding) > 0:
                        batch_embeddings[j] = embedding / np.linalg.norm(embedding)
            
            # Add to dictionary
            for j, text in enumerate(batch_texts):
                embeddings_dict[text] = batch_embeddings[j]
                
        except Exception as e:
            logger.error(f"Error generating embeddings for batch: {e}")
            # Add zero vectors for failed batch
            for text in batch_texts:
                embeddings_dict[text] = np.zeros(model.get_sentence_embedding_dimension())
    
    return embeddings_dict

def get_keywords_embeddings_matrix(
    keywords: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Get embeddings for keywords and return as a matrix.
    
    Args:
        keywords: List of keywords to generate embeddings for
        model_name: Name of the embedding model to use
        batch_size: Number of keywords to process in each batch
        normalize: Whether to normalize the embedding vectors
        show_progress: Whether to show a progress bar
        
    Returns:
        Tuple of (embedding matrix, list of successfully processed keywords)
    """
    embeddings_dict = batch_get_embeddings(
        keywords, 
        model_name=model_name,
        batch_size=batch_size,
        normalize=normalize,
        show_progress=show_progress
    )
    
    # Filter out any keywords that failed to generate embeddings
    successful_keywords = []
    embeddings_list = []
    
    for keyword in keywords:
        embedding = embeddings_dict.get(keyword)
        if embedding is not None and not np.all(embedding == 0):
            successful_keywords.append(keyword)
            embeddings_list.append(embedding)
    
    if len(embeddings_list) == 0:
        logger.warning("No embeddings were successfully generated")
        return np.array([]), []
    
    return np.array(embeddings_list), successful_keywords
