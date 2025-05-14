"""
Evaluation metrics for clustering results.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.metrics import silhouette_score as sk_silhouette_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
import pandas as pd
from collections import Counter

logger = logging.getLogger(__name__)

def silhouette_score(
    embeddings_matrix: np.ndarray,
    labels: Union[List[int], np.ndarray]
) -> float:
    """
    Calculate the silhouette score for a clustering result.
    
    The silhouette score measures how similar an object is to its own cluster
    compared to other clusters. The score ranges from -1 to 1, where:
    - A high value indicates the object is well matched to its own cluster and poorly matched to neighboring clusters
    - A value near 0 indicates the object is on or very close to the decision boundary between two clusters
    - A negative value indicates the object may have been assigned to the wrong cluster
    
    Args:
        embeddings_matrix: Matrix of embeddings
        labels: Cluster labels for each embedding
        
    Returns:
        Silhouette score
    """
    try:
        # Handle case where there's only one cluster
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            logger.warning("Cannot calculate silhouette score with fewer than 2 clusters")
            return 0.0
            
        return sk_silhouette_score(embeddings_matrix, labels)
    except Exception as e:
        logger.error(f"Error calculating silhouette score: {e}")
        return 0.0

def calinski_harabasz_score_wrapper(
    embeddings_matrix: np.ndarray,
    labels: Union[List[int], np.ndarray]
) -> float:
    """
    Calculate the Calinski-Harabasz score for a clustering result.
    
    The Calinski-Harabasz score is defined as the ratio of the between-clusters
    dispersion and the within-cluster dispersion. Higher values indicate better clustering.
    
    Args:
        embeddings_matrix: Matrix of embeddings
        labels: Cluster labels for each embedding
        
    Returns:
        Calinski-Harabasz score
    """
    try:
        # Handle case where there's only one cluster
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            logger.warning("Cannot calculate Calinski-Harabasz score with fewer than 2 clusters")
            return 0.0
            
        return calinski_harabasz_score(embeddings_matrix, labels)
    except Exception as e:
        logger.error(f"Error calculating Calinski-Harabasz score: {e}")
        return 0.0

def davies_bouldin_score_wrapper(
    embeddings_matrix: np.ndarray,
    labels: Union[List[int], np.ndarray]
) -> float:
    """
    Calculate the Davies-Bouldin score for a clustering result.
    
    The Davies-Bouldin score is defined as the average similarity between each
    cluster and its most similar cluster. Lower values indicate better clustering.
    
    Args:
        embeddings_matrix: Matrix of embeddings
        labels: Cluster labels for each embedding
        
    Returns:
        Davies-Bouldin score
    """
    try:
        # Handle case where there's only one cluster
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            logger.warning("Cannot calculate Davies-Bouldin score with fewer than 2 clusters")
            return float('inf')  # Worst possible score
            
        return davies_bouldin_score(embeddings_matrix, labels)
    except Exception as e:
        logger.error(f"Error calculating Davies-Bouldin score: {e}")
        return float('inf')  # Worst possible score

def cluster_size_stats(clusters: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Calculate statistics about cluster sizes.
    
    Args:
        clusters: Dictionary of cluster_id -> list of keywords
        
    Returns:
        Dictionary of statistics
    """
    sizes = [len(keywords) for keywords in clusters.values()]
    
    if not sizes:
        return {
            "total_clusters": 0,
            "total_items": 0,
            "min_size": 0,
            "max_size": 0,
            "mean_size": 0,
            "median_size": 0
        }
    
    stats = {
        "total_clusters": len(clusters),
        "total_items": sum(sizes),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "mean_size": sum(sizes) / len(sizes),
        "median_size": sorted(sizes)[len(sizes) // 2],
        "size_distribution": Counter(sizes)
    }
    
    return stats

def evaluate_clusters(
    clusters: Dict[str, List[str]],
    embeddings_matrix: np.ndarray,
    labels: Union[List[int], np.ndarray]
) -> Dict[str, Any]:
    """
    Evaluate clustering results using multiple metrics.
    
    Args:
        clusters: Dictionary of cluster_id -> list of keywords
        embeddings_matrix: Matrix of embeddings
        labels: Cluster labels for each embedding
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Get cluster size statistics
    size_stats = cluster_size_stats(clusters)
    
    # Calculate clustering quality metrics
    metrics = {
        "silhouette_score": silhouette_score(embeddings_matrix, labels),
        "calinski_harabasz_score": calinski_harabasz_score_wrapper(embeddings_matrix, labels),
        "davies_bouldin_score": davies_bouldin_score_wrapper(embeddings_matrix, labels)
    }
    
    # Combine all metrics
    evaluation = {**size_stats, **metrics}
    
    return evaluation

def compare_clustering_methods(
    methods_results: Dict[str, Tuple[Dict[str, List[str]], np.ndarray, List[str], Any]]
) -> pd.DataFrame:
    """
    Compare different clustering methods based on evaluation metrics.
    
    Args:
        methods_results: Dictionary of method_name -> clustering results tuple
                        (as returned by cluster_keywords function)
        
    Returns:
        DataFrame comparing the performance of different methods
    """
    comparison = []
    
    for method_name, (clusters, embeddings_matrix, keywords, _) in methods_results.items():
        # Convert clusters dictionary to labels array for evaluation
        labels = np.zeros(len(keywords), dtype=int)
        
        for cluster_id, cluster_keywords in clusters.items():
            for keyword in cluster_keywords:
                if keyword in keywords:
                    idx = keywords.index(keyword)
                    try:
                        labels[idx] = int(cluster_id)
                    except ValueError:
                        # Handle non-numeric cluster IDs
                        pass
        
        # Get evaluation metrics
        evaluation = evaluate_clusters(clusters, embeddings_matrix, labels)
        
        # Add method name
        evaluation["method"] = method_name
        
        comparison.append(evaluation)
    
    # Convert to DataFrame
    return pd.DataFrame(comparison)

def evaluate_stability(
    method: str,
    keywords: List[str],
    n_runs: int = 5,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate the stability of a clustering method by running it multiple times.
    
    Args:
        method: Clustering method to use
        keywords: List of keywords to cluster
        n_runs: Number of times to run the clustering
        **kwargs: Additional arguments for the cluster_keywords function
        
    Returns:
        Dictionary of stability metrics
    """
    from .algorithms import cluster_keywords
    
    # Store results for each run
    all_silhouette_scores = []
    all_cluster_counts = []
    all_noise_percentages = []  # For methods that can produce noise points
    
    for i in range(n_runs):
        # Run clustering with different random seed
        random_state = kwargs.get("random_state", 42) + i
        clusters, embeddings_matrix, processed_keywords, _ = cluster_keywords(
            keywords,
            method=method,
            random_state=random_state,
            **kwargs
        )
        
        # Count noise points (cluster ID -1) if present
        noise_count = 0
        if "-1" in clusters:
            noise_count = len(clusters["-1"])
            
        noise_percentage = 100 * noise_count / len(processed_keywords) if processed_keywords else 0
        
        # Create labels array
        labels = []
        for keyword in processed_keywords:
            found = False
            for cluster_id, cluster_keywords in clusters.items():
                if keyword in cluster_keywords:
                    try:
                        labels.append(int(cluster_id))
                    except ValueError:
                        labels.append(0)  # Default for non-numeric IDs
                    found = True
                    break
            if not found:
                labels.append(-1)  # Should not happen, but just in case
        
        # Calculate silhouette score
        try:
            sil_score = silhouette_score(embeddings_matrix, labels)
        except:
            sil_score = 0
            
        all_silhouette_scores.append(sil_score)
        all_cluster_counts.append(len(clusters))
        all_noise_percentages.append(noise_percentage)
    
    # Calculate stability metrics
    stability_metrics = {
        "mean_silhouette": np.mean(all_silhouette_scores),
        "std_silhouette": np.std(all_silhouette_scores),
        "mean_clusters": np.mean(all_cluster_counts),
        "std_clusters": np.std(all_cluster_counts),
        "mean_noise_percentage": np.mean(all_noise_percentages),
        "std_noise_percentage": np.std(all_noise_percentages),
        "n_runs": n_runs
    }
    
    return stability_metrics
