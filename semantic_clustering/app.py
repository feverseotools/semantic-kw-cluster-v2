"""
SemanticKeywordClusterer module for semantic keyword clustering.

This module provides the class-based API for semantic keyword clustering,
making it easier to use the clustering functionality programmatically.
"""

import os
import logging
import json
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile

from .nlp.preprocessing import preprocess_keywords
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
logger = logging.getLogger(__name__)

def load_keywords_from_file(file_path: str) -> List[str]:
    """
    Load keywords from various file formats.
    
    Args:
        file_path: Path to the keywords file (CSV, TXT, JSON)
        
    Returns:
        List of keywords
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []
        
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.csv':
            # Check if file is empty
            if os.path.getsize(file_path) == 0:
                logger.error(f"Empty file: {file_path}")
                return []
                
            # Try to determine if file has headers and use the first column
            df = pd.read_csv(file_path)
            if len(df.columns) > 0:
                # Remove any NaN values
                keywords = df.iloc[:, 0].dropna().astype(str).tolist()
                # Filter out empty strings
                keywords = [k for k in keywords if k.strip()]
            else:
                keywords = []
                
        elif file_ext == '.txt':
            # Check if file is empty
            if os.path.getsize(file_path) == 0:
                logger.error(f"Empty file: {file_path}")
                return []
                
            # Read one keyword per line
            with open(file_path, 'r', encoding='utf-8') as f:
                keywords = [line.strip() for line in f.readlines() if line.strip()]
                
        elif file_ext == '.json':
            # Check if file is empty
            if os.path.getsize(file_path) == 0:
                logger.error(f"Empty file: {file_path}")
                return []
                
            # Try to parse JSON, expecting either an array or an object with a keywords key
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON format in file {file_path}: {e}")
                    return []
                
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
            
        # Additional validation on the keywords
        if not keywords:
            logger.warning(f"No valid keywords found in file: {file_path}")
            return []
            
        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique_keywords.append(k)
        
        # Log the results
        if len(unique_keywords) != len(keywords):
            logger.info(f"Removed {len(keywords) - len(unique_keywords)} duplicate keywords")
        
        logger.info(f"Loaded {len(unique_keywords)} unique keywords from {file_path}")
        return unique_keywords
        
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error in file {file_path}: {e}")
        return []
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error in file {file_path}: {e}. Try saving the file as UTF-8.")
        return []
    except PermissionError as e:
        logger.error(f"Permission denied when accessing file {file_path}: {e}")
        return []
    except MemoryError as e:
        logger.error(f"Not enough memory to load file {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading keywords from {file_path}: {e}")
        return []

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
        perform_preprocessing: bool = True,
        eps: float = 0.5,  # For DBSCAN
        min_samples: int = 5,  # For DBSCAN/HDBSCAN
        min_cluster_size: int = 5,  # For HDBSCAN
        random_state: int = 42  # For reproducibility
    ):
        """
        Initialize the clusterer.
        
        Args:
            embedding_model: Name of the embedding model to use
            method: Clustering method to use
            n_clusters: Number of clusters (optional)
            perform_preprocessing: Whether to preprocess keywords
            eps: Epsilon parameter for DBSCAN
            min_samples: Minimum samples parameter for DBSCAN/HDBSCAN
            min_cluster_size: Minimum cluster size for HDBSCAN
            random_state: Random state for reproducibility
        """
        self.embedding_model = embedding_model
        self.method = method
        self.n_clusters = n_clusters
        self.perform_preprocessing = perform_preprocessing
        self.eps = eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        
        self.clusters = {}
        self.cluster_labels = {}
        self.metrics = {}
        self.embeddings_2d = None
        self.labels = None
        self.keywords = []
        self.processed_keywords = []
        self.valid_keywords = []
        self.visualization_method = "umap"
        
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
        # Prepare parameters for the clustering method
        kwargs = {
            "random_state": self.random_state
        }
        
        # Add method-specific parameters
        if self.method == "dbscan":
            kwargs["eps"] = self.eps
            kwargs["min_samples"] = self.min_samples
        elif self.method == "hdbscan":
            kwargs["min_cluster_size"] = self.min_cluster_size
            if self.min_samples:
                kwargs["min_samples"] = self.min_samples
        
        # Get embeddings
        embeddings_matrix, self.valid_keywords = get_keywords_embeddings_matrix(
            self.processed_keywords,
            model_name=self.embedding_model,
            show_progress=True
        )
        
        self.logger.info(f"Generated embeddings for {len(self.valid_keywords)} keywords")
        
        # Optimize number of clusters if requested
        if optimize and self.method in ['kmeans', 'agglomerative']:
            if len(self.valid_keywords) < min_clusters:
                min_clusters = max(2, len(self.valid_keywords) // 2)
                max_clusters = min(max_clusters, len(self.valid_keywords) - 1)
                
            self.logger.info(f"Optimizing number of clusters between {min_clusters} and {max_clusters}")
            opt_n_clusters, best_score, _ = optimize_clusters(
                self.valid_keywords,
                method=self.method,
                embeddings_matrix=embeddings_matrix,
                embedding_model=self.embedding_model,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                random_state=self.random_state
            )
            self.n_clusters = opt_n_clusters
            self.logger.info(f"Optimal number of clusters: {self.n_clusters} (score: {best_score:.4f})")
        
        # Perform clustering
        self.clusters, _, _, cluster_model = cluster_keywords(
            self.valid_keywords,
            method=self.method,
            n_clusters=self.n_clusters,
            embeddings_matrix=embeddings_matrix,
            embedding_model=self.embedding_model,
            **kwargs
        )
        
        self.logger.info(f"Created {len(self.clusters)} clusters")
        
        # Extract cluster labels
        self.cluster_labels = extract_cluster_labels(
            self.clusters,
            method=label_method
        )
        
        # Get cluster labels as a list for evaluation
        if self.method in ["kmeans", "dbscan", "hdbscan", "agglomerative"]:
            self.labels = cluster_model.labels_
        else:
            # Create labels from clusters dictionary
            self.labels = []
            for i, keyword in enumerate(self.valid_keywords):
                label = -1
                for cluster_id, cluster_keywords in self.clusters.items():
                    if keyword in cluster_keywords:
                        try:
                            label = int(cluster_id)
                        except ValueError:
                            label = 0
                        break
                self.labels.append(label)
        
        # Evaluate clusters
        self.metrics = evaluate_clusters(self.clusters, embeddings_matrix, self.labels)
        
        # Reduce dimensionality for visualization
        self.embeddings_2d = dimensionality_reduction(
            embeddings_matrix,
            method=self.visualization_method,
            n_components=2
        )
        
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
                    self.clusters,
                    output_path,
                    cluster_labels=self.cluster_labels,
                    evaluation_metrics=self.metrics
                )
                if success:
                    output_files['json'] = output_path
                    
            elif fmt.lower() == 'excel':
                output_path = os.path.join(output_dir, f"{file_prefix}_{timestamp}.xlsx")
                success = export_to_excel(
                    self.clusters,
                    output_path,
                    cluster_labels=self.cluster_labels,
                    evaluation_metrics=self.metrics
                )
                if success:
                    output_files['excel'] = output_path
                    
            elif fmt.lower() == 'html':
                output_path = os.path.join(output_dir, f"{file_prefix}_{timestamp}.html")
                success = export_to_html(
                    self.clusters,
                    output_path,
                    cluster_labels=self.cluster_labels,
                    evaluation_metrics=self.metrics,
                    embeddings_2d=self.embeddings_2d,
                    labels=self.labels
                )
                if success:
                    output_files['html'] = output_path
                    
            elif fmt.lower() == 'pdf':
                output_path = os.path.join(output_dir, f"{file_prefix}_{timestamp}.pdf")
                success = export_to_pdf(
                    self.clusters,
                    output_path,
                    cluster_labels=self.cluster_labels,
                    evaluation_metrics=self.metrics,
                    embeddings_2d=self.embeddings_2d,
                    labels=self.labels
                )
                if success:
                    output_files['pdf'] = output_path
                    
            else:
                self.logger.warning(f"Unsupported export format: {fmt}")
        
        return output_files
    
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
    
    def set_visualization_method(self, method: str) -> None:
        """
        Set the dimensionality reduction method for visualization.
        
        Args:
            method: Dimensionality reduction method ('pca' or 'umap')
        """
        if method.lower() not in ['pca', 'umap']:
            self.logger.warning(f"Unsupported visualization method: {method}. Using 'umap' instead.")
            method = 'umap'
        
        self.visualization_method = method.lower()
    
    def get_visualization_data(self) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
        """
        Get data for visualization.
        
        Returns:
            Tuple of (2D embeddings, cluster labels)
        """
        return self.embeddings_2d, self.labels
    
    def run_pipeline(
        self,
        keywords: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        optimize: bool = False,
        min_clusters: int = 2,
        max_clusters: int = 20,
        output_dir: str = 'output',
        formats: List[str] = ['json'],
        file_prefix: str = 'clusters',
        label_method: str = 'tfidf'
    ) -> Dict[str, Any]:
        """
        Run the complete clustering pipeline.
        
        Args:
            keywords: List of keywords (optional)
            file_path: Path to keywords file (optional)
            optimize: Whether to optimize the number of clusters
            min_clusters: Minimum number of clusters for optimization
            max_clusters: Maximum number of clusters for optimization
            output_dir: Directory to save the output files
            formats: List of output formats
            file_prefix: Prefix for the output files
            label_method: Method for extracting cluster labels
            
        Returns:
            Dictionary with results and output files
        """
        # Load keywords
        if keywords:
            self.load_keywords(keywords)
        elif file_path:
            self.load_keywords_from_file(file_path)
        
        if not self.processed_keywords:
            self.logger.error("No keywords provided")
            return {"success": False, "error": "No keywords provided"}
        
        # Perform clustering
        self.cluster(
            optimize=optimize,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            label_method=label_method
        )
        
        # Save results
        output_files = self.save(
            output_dir=output_dir,
            formats=formats,
            file_prefix=file_prefix
        )
        
        return {
            "success": True,
            "clusters": self.clusters,
            "cluster_labels": self.cluster_labels,
            "metrics": self.metrics,
            "output_files": output_files
        }

def main():
    """
    Command line interface for semantic keyword clustering.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Semantic Keyword Clustering")
    
    # Input options
    parser.add_argument(
        "--input", "-i", 
        type=str,
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
    
    parser.add_argument(
        "--visualization",
        type=str,
        choices=["pca", "umap"],
        default="umap",
        help="Visualization method for dimensionality reduction"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create clusterer
    clusterer = SemanticKeywordClusterer(
        embedding_model=args.embedding_model,
        method=args.method,
        n_clusters=args.clusters,
        perform_preprocessing=not args.no_preprocess
    )
    
    # Set visualization method
    clusterer.set_visualization_method(args.visualization)
    
    # Run pipeline
    if args.input:
        results = clusterer.run_pipeline(
            file_path=args.input,
            optimize=args.optimize,
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters,
            output_dir=args.output_dir,
            formats=args.formats,
            file_prefix=args.prefix
        )
        
        if results["success"]:
            print(f"Clustering completed successfully")
            print(f"Created {len(results['clusters'])} clusters")
            
            metrics = results.get("metrics", {})
            if "silhouette_score" in metrics:
                print(f"Silhouette score: {metrics['silhouette_score']:.4f}")
            
            print("Results saved to:")
            for fmt, path in results.get("output_files", {}).items():
                print(f"  {fmt.upper()}: {path}")
            
            return 0
        else:
            print(f"Error: {results.get('error', 'Unknown error')}")
            return 1
    else:
        print("No input file specified")
        return 1

if __name__ == "__main__":
    main()
