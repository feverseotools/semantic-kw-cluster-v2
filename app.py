#!/usr/bin/env python3
"""
Main application for semantic keyword clustering.

This module provides the main functionality for the semantic keyword clustering
application, including CLI interface and core workflow.
"""

import os
import sys
import logging
import argparse
from typing import List, Dict, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import json

from .nlp.preprocessing import preprocess_text, preprocess_keywords
from .nlp.intent import classify_search_intent
from .nlp.models import get_embedding_model, list_available_models

from .clustering.embeddings import get_keywords_embeddings_matrix
from .clustering.algorithms import (
    cluster_keywords, 
    optimize_clusters, 
    extract_cluster_labels,
    dimensionality_reduction
)
from .clustering.evaluation import evaluate_clusters

from .export.pdf import export_to_pdf
from .export.excel import export_to_excel
from .export.html import export_to_html
from .export.json import export_to_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def load_keywords_from_file(file_path: str) -> List[str]:
    """
    Load keywords from various file formats.
    
    Args:
        file_path: Path to the keywords file (CSV, TXT, JSON)
        
    Returns:
        List of keywords
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.csv':
            # Try to determine if file has headers and use the first column
            df = pd.read_csv(file_path)
            if len(df.columns) > 0:
                # Use first column as keywords
                keywords = df.iloc[:, 0].astype(str).tolist()
            else:
                keywords = []
                
        elif file_ext == '.txt':
            # Read one keyword per line
            with open(file_path, 'r', encoding='utf-8') as f:
                keywords = [line.strip() for line in f.readlines() if line.strip()]
                
        elif file_ext == '.json':
            # Try to parse JSON, expecting either an array or an object with a keywords key
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                keywords = [str(k) for k in data if k]
            elif isinstance(data, dict) and 'keywords' in data:
                keywords = [str(k) for k in data['keywords'] if k]
            else:
                logger.error(f"Could not find keywords in JSON file: {file_path}")
                keywords = []
                
        else:
            logger.error(f"Unsupported file format: {file_ext}")
            keywords = []
            
        logger.info(f"Loaded {len(keywords)} keywords from {file_path}")
        return keywords
        
    except Exception as e:
        logger.error(f"Error loading keywords from {file_path}: {e}")
        return []

def save_clusters(
    clusters: Dict[str, List[str]],
    output_dir: str,
    file_prefix: str,
    formats: List[str] = ['json'],
    cluster_labels: Optional[Dict[str, str]] = None,
    evaluation_metrics: Optional[Dict[str, Any]] = None,
    embeddings_2d: Optional[np.ndarray] = None,
    labels: Optional[List[int]] = None
) -> Dict[str, str]:
    """
    Save clusters to various file formats.
    
    Args:
        clusters: Dictionary of cluster_id -> list of keywords
        output_dir: Directory to save the files
        file_prefix: Prefix for the output files
        formats: List of formats to save ('json', 'excel', 'html', 'pdf')
        cluster_labels: Dictionary of cluster_id -> descriptive label (optional)
        evaluation_metrics: Dictionary of evaluation metrics (optional)
        embeddings_2d: 2D embeddings for visualization (optional)
        labels: Cluster labels corresponding to embeddings_2d (optional)
        
    Returns:
        Dictionary of format -> output file path
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_files = {}
    
    # Export to each requested format
    for fmt in formats:
        if fmt.lower() == 'json':
            output_path = os.path.join(output_dir, f"{file_prefix}_{timestamp}.json")
            success = export_to_json(
                clusters,
                output_path,
                cluster_labels=cluster_labels,
                evaluation_metrics=evaluation_metrics
            )
            if success:
                output_files['json'] = output_path
                
        elif fmt.lower() == 'excel':
            output_path = os.path.join(output_dir, f"{file_prefix}_{timestamp}.xlsx")
            success = export_to_excel(
                clusters,
                output_path,
                cluster_labels=cluster_labels,
                evaluation_metrics=evaluation_metrics
            )
            if success:
                output_files['excel'] = output_path
                
        elif fmt.lower() == 'html':
            output_path = os.path.join(output_dir, f"{file_prefix}_{timestamp}.html")
            success = export_to_html(
                clusters,
                output_path,
                cluster_labels=cluster_labels,
                evaluation_metrics=evaluation_metrics,
                embeddings_2d=embeddings_2d,
                labels=labels
            )
            if success:
                output_files['html'] = output_path
                
        elif fmt.lower() == 'pdf':
            output_path = os.path.join(output_dir, f"{file_prefix}_{timestamp}.pdf")
            success = export_to_pdf(
                clusters,
                output_path,
                cluster_labels=cluster_labels,
                evaluation_metrics=evaluation_metrics,
                embeddings_2d=embeddings_2d,
                labels=labels
            )
            if success:
                output_files['pdf'] = output_path
                
        else:
            logger.warning(f"Unsupported export format: {fmt}")
    
    return output_files

def run_clustering_pipeline(
    keywords: List[str],
    n_clusters: Optional[int] = None,
    method: str = 'kmeans',
    label_method: str = 'tfidf',
    embedding_model: str = 'all-MiniLM-L6-v2',
    optimize: bool = False,
    min_clusters: int = 2,
    max_clusters: int = 20,
    visualization_dimensions: int = 2,
    visualization_method: str = 'umap',
    preprocess: bool = True
) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, Any], np.ndarray, List[int]]:
    """
    Run the complete clustering pipeline.
    
    Args:
        keywords: List of keywords to cluster
        n_clusters: Number of clusters (optional, will be optimized if not provided)
        method: Clustering method ('kmeans', 'dbscan', 'hdbscan', 'agglomerative')
        label_method: Method for extracting cluster labels ('frequent', 'tfidf', 'centroid')
        embedding_model: Name of the embedding model to use
        optimize: Whether to optimize the number of clusters
        min_clusters: Minimum number of clusters for optimization
        max_clusters: Maximum number of clusters for optimization
        visualization_dimensions: Number of dimensions for visualization
        visualization_method: Method for dimensionality reduction ('pca', 'umap')
        preprocess: Whether to preprocess the keywords
        
    Returns:
        Tuple of (clusters dict, cluster labels dict, evaluation metrics dict, 
                 2D embeddings, cluster labels)
    """
    # Log configuration
    logger.info(f"Starting clustering pipeline with {len(keywords)} keywords")
    logger.info(f"Clustering method: {method}")
    logger.info(f"Embedding model: {embedding_model}")
    
    # Preprocess keywords if requested
    if preprocess:
        processed_keywords = preprocess_keywords(keywords)
        logger.info(f"Preprocessing complete. {len(processed_keywords)} keywords after preprocessing")
    else:
        processed_keywords = keywords
    
    # Get embeddings
    embeddings_matrix, valid_keywords = get_keywords_embeddings_matrix(
        processed_keywords, 
        model_name=embedding_model,
        show_progress=True
    )
    
    logger.info(f"Generated embeddings for {len(valid_keywords)} keywords")
    
    # Optimize number of clusters if requested
    if optimize or n_clusters is None:
        if method in ['kmeans', 'agglomerative']:
            logger.info(f"Optimizing number of clusters between {min_clusters} and {max_clusters}")
            opt_n_clusters, best_score, _ = optimize_clusters(
                valid_keywords,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                method=method,
                embedding_model=embedding_model
            )
            n_clusters = opt_n_clusters
            logger.info(f"Optimal number of clusters: {n_clusters} (score: {best_score:.4f})")
        else:
            logger.info(f"Cluster optimization not supported for method: {method}")
    
    # Perform clustering
    clusters, _, clusters_keywords, cluster_model = cluster_keywords(
        valid_keywords,
        method=method,
        n_clusters=n_clusters,
        embeddings_matrix=embeddings_matrix,
        embedding_model=embedding_model
    )
    
    logger.info(f"Created {len(clusters)} clusters")
    
    # Extract cluster labels
    cluster_labels = extract_cluster_labels(
        clusters,
        method=label_method
    )
    
    # Get cluster labels as a list for evaluation
    if method == 'kmeans':
        labels = cluster_model.labels_
    elif method == 'dbscan':
        labels = cluster_model.labels_
    elif method == 'hdbscan':
        labels = cluster_model.labels_
    elif method == 'agglomerative':
        labels = cluster_model.labels_
    else:
        # Create labels from clusters dictionary
        labels = []
        for i, keyword in enumerate(valid_keywords):
            label = -1
            for cluster_id, cluster_keywords in clusters.items():
                if keyword in cluster_keywords:
                    try:
                        label = int(cluster_id)
                    except ValueError:
                        label = 0
                    break
            labels.append(label)
    
    # Evaluate clusters
    metrics = evaluate_clusters(clusters, embeddings_matrix, labels)
    logger.info(f"Silhouette score: {metrics.get('silhouette_score', 0):.4f}")
    
    # Reduce dimensionality for visualization
    embeddings_2d = dimensionality_reduction(
        embeddings_matrix,
        method=visualization_method,
        n_components=visualization_dimensions
    )
    
    logger.info("Clustering pipeline complete")
    
    return clusters, cluster_labels, metrics, embeddings_2d, labels

def main():
    """
    Main function for CLI execution.
    """
    parser = argparse.ArgumentParser(description="Semantic Keyword Clustering")
    
    # Input options
    parser.add_argument(
        "--input", "-i", 
        type=str,
        required=True,
        help="Path to input file (CSV, TXT, JSON)"
    )
    
    # Clustering options
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["kmeans", "dbscan", "hdbscan", "agglomerative"],
        default="kmeans",
        help="Clustering method to use"
    )
    
    parser.add_argument(
        "--clusters", "-c",
        type=int,
        default=None,
        help="Number of clusters (will be optimized if not specified)"
    )
    
    parser.add_argument(
        "--optimize", "-o",
        action="store_true",
        help="Optimize number of clusters"
    )
    
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=2,
        help="Minimum number of clusters for optimization"
    )
    
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=20,
        help="Maximum number of clusters for optimization"
    )
    
    # Embedding options
    parser.add_argument(
        "--embedding-model", "-e",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model to use"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available embedding models and exit"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", "-d",
        type=str,
        default="output",
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--formats", "-f",
        type=str,
        nargs="+",
        choices=["json", "excel", "html", "pdf"],
        default=["json"],
        help="Output formats"
    )
    
    parser.add_argument(
        "--prefix", "-p",
        type=str,
        default="clusters",
        help="Prefix for output files"
    )
    
    # Miscellaneous
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip keyword preprocessing"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # List models and exit if requested
    if args.list_models:
        models = list_available_models()
        print("Available embedding models:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        return 0
    
    # Load keywords
    keywords = load_keywords_from_file(args.input)
    
    if not keywords:
        logger.error("No keywords found in input file")
        return 1
    
    # Run clustering pipeline
    clusters, cluster_labels, metrics, embeddings_2d, labels = run_clustering_pipeline(
        keywords,
        n_clusters=args.clusters,
        method=args.method,
        embedding_model=args.embedding_model,
        optimize=args.optimize,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        preprocess=not args.no_preprocess
    )
    
    # Save results
    output_files = save_clusters(
        clusters,
        args.output_dir,
        args.prefix,
        formats=args.formats,
        cluster_labels=cluster_labels,
        evaluation_metrics=metrics,
        embeddings_2d=embeddings_2d,
        labels=labels
    )
    
    logger.info("Results saved to:")
    for fmt, path in output_files.items():
        logger.info(f"  {fmt.upper()}: {path}")
    
    return 0

class SemanticKeywordClusterer:
    """
    Class-based API for semantic keyword clustering.
    
    This class provides a more object-oriented interface to the clustering
    functionality, useful for integration with other applications.
    """
    
    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        method: str = 'kmeans',
        n_clusters: Optional[int] = None,
        perform_preprocessing: bool = True
    ):
        """
        Initialize the clusterer.
        
        Args:
            embedding_model: Name of the embedding model to use
            method: Clustering method to use
            n_clusters: Number of clusters (optional)
            perform_preprocessing: Whether to preprocess keywords
        """
        self.embedding_model = embedding_model
        self.method = method
        self.n_clusters = n_clusters
        self.perform_preprocessing = perform_preprocessing
        
        self.clusters = {}
        self.cluster_labels = {}
        self.metrics = {}
        self.embeddings_2d = None
        self.labels = None
        self.keywords = []
        self.processed_keywords = []
        self.valid_keywords = []
        
        self.logger = logging.getLogger(f"{__name__}.SemanticKeywordClusterer")
    
    def load_keywords(self, keywords: List[str]) -> int:
        """
        Load keywords into the clusterer.
        
        Args:
            keywords: List of keywords
            
        Returns:
            Number of loaded keywords
        """
        self.keywords = keywords
        
        if self.perform_preprocessing:
            self.processed_keywords = preprocess_keywords(self.keywords)
        else:
            self.processed_keywords = self.keywords
            
        self.logger.info(f"Loaded {len(self.processed_keywords)} keywords")
        return len(self.processed_keywords)
    
    def load_keywords_from_file(self, file_path: str) -> int:
        """
        Load keywords from a file.
        
        Args:
            file_path: Path to the keywords file (CSV, TXT, JSON)
            
        Returns:
            Number of loaded keywords
        """
        keywords = load_keywords_from_file(file_path)
        return self.load_keywords(keywords)
    
    def cluster(
        self,
        optimize: bool = False,
        min_clusters: int = 2,
        max_clusters: int = 20,
        label_method: str = 'tfidf'
    ) -> Dict[str, List[str]]:
        """
        Perform clustering on the loaded keywords.
        
        Args:
            optimize: Whether to optimize the number of clusters
            min_clusters: Minimum number of clusters for optimization
            max_clusters: Maximum number of clusters for optimization
            label_method: Method for extracting cluster labels
            
        Returns:
            Dictionary of cluster_id -> list of keywords
        """
        result = run_clustering_pipeline(
            self.processed_keywords,
            n_clusters=self.n_clusters,
            method=self.method,
            label_method=label_method,
            embedding_model=self.embedding_model,
            optimize=optimize,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            preprocess=False  # Already preprocessed if needed
        )
        
        self.clusters, self.cluster_labels, self.metrics, self.embeddings_2d, self.labels = result
        self.valid_keywords = []
        
        # Collect valid keywords from clusters
        for keywords in self.clusters.values():
            self.valid_keywords.extend(keywords)
        
        return self.clusters
    
    def save(
        self,
        output_dir: str,
        formats: List[str] = ['json'],
        file_prefix: str = 'clusters'
    ) -> Dict[str, str]:
        """
        Save clustering results to files.
        
        Args:
            output_dir: Directory to save the files
            formats: List of formats to save
            file_prefix: Prefix for the output files
            
        Returns:
            Dictionary of format -> output file path
        """
        return save_clusters(
            self.clusters,
            output_dir,
            file_prefix,
            formats=formats,
            cluster_labels=self.cluster_labels,
            evaluation_metrics=self.metrics,
            embeddings_2d=self.embeddings_2d,
            labels=self.labels
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get clustering evaluation metrics.
        
        Returns:
            Dictionary of evaluation metrics
        """
        return self.metrics
    
    def get_cluster_labels(self) -> Dict[str, str]:
        """
        Get descriptive labels for clusters.
        
        Returns:
            Dictionary of cluster_id -> descriptive label
        """
        return self.cluster_labels
    
    def get_visualization_data(self) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
        """
        Get data for visualization.
        
        Returns:
            Tuple of (2D embeddings, cluster labels)
        """
        return self.embeddings_2d, self.labels

if __name__ == "__main__":
    sys.exit(main())
