"""
Utility functions for semantic keyword clustering.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def batch_process_large_dataset(
    keywords: List[str],
    batch_size: int = 1000,
    processing_func: Callable = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Process large keyword datasets in batches to reduce memory usage.
    
    Args:
        keywords: Full list of keywords
        batch_size: Size of each batch
        processing_func: Function to process each batch
        **kwargs: Additional arguments for the processing function
        
    Returns:
        Combined processing results
    """
    if processing_func is None:
        from semantic_clustering.clustering.algorithms import cluster_keywords
        processing_func = cluster_keywords
    
    if len(keywords) <= batch_size:
        # If the dataset is smaller than the batch size, process everything at once
        return processing_func(keywords, **kwargs)
    
    logger.info(f"Processing {len(keywords)} keywords in batches of {batch_size}")
    
    # Initialize result containers
    all_results = {}
    all_embeddings = []
    all_labels = []
    cluster_offset = 0
    
    # Process keywords in batches
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} keywords)")
        
        # Process this batch
        batch_results = processing_func(batch, **kwargs)
        
        # Handle different return types
        if isinstance(batch_results, tuple) and len(batch_results) >= 2:
            # Assume return format is (clusters, embeddings_matrix, keywords, model)
            batch_clusters = batch_results[0]
            batch_embeddings = batch_results[1]
            
            # Adjust cluster IDs to avoid conflicts
            adjusted_clusters = {}
            for cluster_id, batch_keywords in batch_clusters.items():
                try:
                    new_id = str(int(cluster_id) + cluster_offset)
                except ValueError:
                    # Handle non-numeric cluster IDs
                    new_id = f"{cluster_id}_{i//batch_size}"
                adjusted_clusters[new_id] = batch_keywords
            
            # Update offset for next batch
            cluster_offset += len(batch_clusters)
            
            # Update results
            if not all_results:
                all_results = [adjusted_clusters, batch_embeddings, batch_results[2], batch_results[3]]
            else:
                all_results[0].update(adjusted_clusters)
                all_results[1] = np.vstack([all_results[1], batch_embeddings]) if all_results[1].size > 0 else batch_embeddings
                all_results[2].extend(batch_results[2])
                # Model from the first batch is used
                
        elif isinstance(batch_results, dict):
            # Handle dictionary return format
            if 'clusters' in batch_results:
                # Adjust cluster IDs to avoid conflicts
                adjusted_clusters = {}
                for cluster_id, batch_keywords in batch_results['clusters'].items():
                    try:
                        new_id = str(int(cluster_id) + cluster_offset)
                    except ValueError:
                        # Handle non-numeric cluster IDs
                        new_id = f"{cluster_id}_{i//batch_size}"
                    adjusted_clusters[new_id] = batch_keywords
                
                # Update offset for next batch
                cluster_offset += len(batch_results['clusters'])
                
                # Update results
                if 'clusters' not in all_results:
                    all_results['clusters'] = {}
                all_results['clusters'].update(adjusted_clusters)
                
                # Copy other keys
                for key, value in batch_results.items():
                    if key != 'clusters':
                        all_results[key] = value
            else:
                # Simple merge of dictionaries
                for key, value in batch_results.items():
                    if key not in all_results:
                        all_results[key] = value
                    elif isinstance(value, list):
                        all_results[key].extend(value)
                    elif isinstance(value, dict):
                        all_results[key].update(value)
                    else:
                        all_results[key] = value
    
    logger.info(f"Batch processing complete")
    return all_results


def memory_efficient_embedding(
    keywords: List[str],
    embedding_func: Callable,
    batch_size: int = 500,
    **kwargs
) -> np.ndarray:
    """
    Generate embeddings for a large list of keywords in a memory-efficient way.
    
    Args:
        keywords: List of keywords to embed
        embedding_func: Function to generate embeddings
        batch_size: Size of each batch
        **kwargs: Additional arguments for the embedding function
        
    Returns:
        Array of embeddings
    """
    if len(keywords) <= batch_size:
        return embedding_func(keywords, **kwargs)
    
    all_embeddings = []
    
    # Process in batches
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i+batch_size]
        logger.info(f"Embedding batch {i//batch_size + 1} ({len(batch)} keywords)")
        
        batch_embeddings = embedding_func(batch, **kwargs)
        all_embeddings.append(batch_embeddings)
    
    # Combine results
    if all(isinstance(emb, np.ndarray) for emb in all_embeddings):
        return np.vstack(all_embeddings)
    else:
        # Handle other return types
        combined = []
        for emb in all_embeddings:
            if isinstance(emb, list):
                combined.extend(emb)
            else:
                combined.append(emb)
        return combined
