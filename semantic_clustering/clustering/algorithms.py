"""
Clustering algorithms for semantic keyword clustering.
"""

import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
import logging
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import umap
import hdbscan
from collections import Counter
from tqdm import tqdm

from .embeddings import get_keywords_embeddings_matrix
from ..nlp.preprocessing import preprocess_text

logger = logging.getLogger(__name__)

def cluster_keywords(
    keywords: List[str],
    method: str = "kmeans",
    n_clusters: Optional[int] = None,
    embeddings_matrix: Optional[np.ndarray] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    random_state: int = 42,
    **kwargs
) -> Tuple[Dict[str, List[str]], np.ndarray, List[str], Optional[Any]]:
    """
    Cluster keywords based on their semantic embeddings.
    
    Args:
        keywords: List of keywords to cluster
        method: Clustering method to use ('kmeans', 'dbscan', 'hdbscan', 'agglomerative')
        n_clusters: Number of clusters (required for kmeans and agglomerative)
        embeddings_matrix: Pre-computed embeddings matrix (optional)
        embedding_model: Name of the embedding model to use if embeddings_matrix is not provided
        random_state: Random state for reproducibility
        **kwargs: Additional arguments for the clustering algorithm
        
    Returns:
        Tuple of (clusters dict, embeddings matrix, processed keywords, clustering model)
    """
    # Get embeddings if not provided
    if embeddings_matrix is None:
        embeddings_matrix, processed_keywords = get_keywords_embeddings_matrix(
            keywords, model_name=embedding_model
        )
    else:
        processed_keywords = keywords
        
    if len(processed_keywords) == 0:
        logger.error("No valid keywords to cluster")
        return {}, np.array([]), [], None
        
    # Determine number of clusters if not provided for kmeans
    if method == "kmeans" and n_clusters is None:
        n_clusters = min(int(np.sqrt(len(processed_keywords))), 100)
        logger.info(f"Automatically determined n_clusters: {n_clusters}")
    
    # Apply the selected clustering algorithm
    cluster_model = None
    
    if method == "kmeans":
        cluster_model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            **kwargs
        )
        labels = cluster_model.fit_predict(embeddings_matrix)
        
    elif method == "dbscan":
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 5)
        cluster_model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        labels = cluster_model.fit_predict(embeddings_matrix)
        
    elif method == "hdbscan":
        min_cluster_size = kwargs.get("min_cluster_size", 5)
        min_samples = kwargs.get("min_samples", None)
        cluster_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            **kwargs
        )
        labels = cluster_model.fit_predict(embeddings_matrix)
        
    elif method == "agglomerative":
        linkage = kwargs.get("linkage", "ward")
        cluster_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            **kwargs
        )
        labels = cluster_model.fit_predict(embeddings_matrix)
        
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    # Organize keywords into clusters
    clusters = {}
    for i, label in enumerate(labels):
        label_str = str(label)
        if label_str not in clusters:
            clusters[label_str] = []
        clusters[label_str].append(processed_keywords[i])
    
    return clusters, embeddings_matrix, processed_keywords, cluster_model

def optimize_clusters(
    keywords: List[str],
    min_clusters: int = 2,
    max_clusters: int = 20,
    method: str = "kmeans",
    embedding_model: str = "all-MiniLM-L6-v2",
    random_state: int = 42
) -> Tuple[int, float, List[Tuple[int, float]]]:
    """
    Find the optimal number of clusters using silhouette score.
    
    Args:
        keywords: List of keywords to cluster
        min_clusters: Minimum number of clusters to try
        max_clusters: Maximum number of clusters to try
        method: Clustering method ('kmeans' or 'agglomerative')
        embedding_model: Name of the embedding model to use
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (optimal number of clusters, best silhouette score, list of all scores)
    """
    # Get embeddings once to reuse for all cluster counts
    embeddings_matrix, processed_keywords = get_keywords_embeddings_matrix(
        keywords, model_name=embedding_model
    )
    
    if len(processed_keywords) < min_clusters:
        logger.warning(f"Not enough valid keywords ({len(processed_keywords)}) for minimum clusters ({min_clusters})")
        min_clusters = max(2, len(processed_keywords) // 2)
        max_clusters = min(max_clusters, len(processed_keywords) - 1)
    
    scores = []
    best_score = -1
    best_n_clusters = min_clusters
    
    # Try different numbers of clusters
    for n_clusters in tqdm(range(min_clusters, min(max_clusters+1, len(processed_keywords))), desc="Optimizing clusters"):
        try:
            _, _, _, cluster_model = cluster_keywords(
                processed_keywords,
                method=method,
                n_clusters=n_clusters,
                embeddings_matrix=embeddings_matrix,
                embedding_model=embedding_model,
                random_state=random_state
            )
            
            if method == "kmeans":
                labels = cluster_model.labels_
            elif method == "agglomerative":
                labels = cluster_model.labels_
            else:
                raise ValueError(f"Method {method} not supported for optimization")
                
            score = silhouette_score(embeddings_matrix, labels)
            scores.append((n_clusters, score))
            
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                
        except Exception as e:
            logger.error(f"Error optimizing for {n_clusters} clusters: {e}")
    
    return best_n_clusters, best_score, scores

def extract_cluster_labels(
    clusters: Dict[str, List[str]],
    n_words: int = 3,
    method: str = "tfidf"
) -> Dict[str, str]:
    """
    Extract descriptive labels for each cluster.
    
    Args:
        clusters: Dictionary of cluster_id -> list of keywords
        n_words: Number of words to include in each label
        method: Method to use for label extraction ('frequent', 'tfidf', 'centroid')
        
    Returns:
        Dictionary mapping cluster IDs to their descriptive labels
    """
    cluster_labels = {}
    
    for cluster_id, keywords in clusters.items():
        if not keywords:
            cluster_labels[cluster_id] = f"Cluster {cluster_id}"
            continue
            
        if method == "frequent":
            # Use most frequent words in the cluster
            all_words = []
            for keyword in keywords:
                processed = preprocess_text(keyword)
                all_words.extend(processed.split())
                
            word_counts = Counter(all_words)
            most_common = word_counts.most_common(n_words)
            top_words = [word for word, _ in most_common]
            
            if top_words:
                cluster_labels[cluster_id] = " / ".join(top_words)
            else:
                cluster_labels[cluster_id] = f"Cluster {cluster_id}"
                
        elif method == "tfidf":
            # Simple implementation of TF-IDF-like scoring
            # Count term frequency within cluster
            cluster_term_freq = Counter()
            for keyword in keywords:
                processed = preprocess_text(keyword)
                for word in processed.split():
                    cluster_term_freq[word] += 1
                    
            # Count document frequency across all clusters
            doc_freq = Counter()
            for c_id, c_keywords in clusters.items():
                doc_words = set()
                for keyword in c_keywords:
                    processed = preprocess_text(keyword)
                    doc_words.update(processed.split())
                for word in doc_words:
                    doc_freq[word] += 1
            
            # Calculate TF-IDF scores
            n_clusters = len(clusters)
            tfidf_scores = {}
            for word, freq in cluster_term_freq.items():
                if doc_freq[word] > 0:
                    tfidf_scores[word] = freq * np.log(n_clusters / doc_freq[word])
            
            # Get top words by TF-IDF
            top_words = [word for word, _ in sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:n_words]]
            
            if top_words:
                cluster_labels[cluster_id] = " / ".join(top_words)
            else:
                cluster_labels[cluster_id] = f"Cluster {cluster_id}"
                
        elif method == "centroid":
            # Just use the keywords closest to the centroid
            # (Simplified version - just takes the shortest keywords)
            sorted_keywords = sorted(keywords, key=len)
            label_words = sorted_keywords[:n_words]
            cluster_labels[cluster_id] = " / ".join(label_words)
            
        else:
            cluster_labels[cluster_id] = f"Cluster {cluster_id}"
    
    return cluster_labels

def dimensionality_reduction(
    embeddings_matrix: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    random_state: int = 42,
    **kwargs
) -> np.ndarray:
    """
    Reduce the dimensionality of embeddings for visualization.
    
    Args:
        embeddings_matrix: Matrix of embeddings
        method: Method to use ('pca' or 'umap')
        n_components: Number of dimensions to reduce to
        random_state: Random state for reproducibility
        **kwargs: Additional arguments for the reduction method
        
    Returns:
        Reduced embeddings matrix
    """
    if embeddings_matrix.shape[0] == 0:
        return np.array([])
        
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(embeddings_matrix)
        
    elif method == "umap":
        n_neighbors = kwargs.get("n_neighbors", 15)
        min_dist = kwargs.get("min_dist", 0.1)
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            **kwargs
        )
        return reducer.fit_transform(embeddings_matrix)
        
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")
