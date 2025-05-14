"""
Clustering Module Test Suite for Semantic Keyword Clustering.

This module contains comprehensive unit and integration tests
for clustering algorithms, embeddings, and evaluation metrics.
"""

import os
import numpy as np
import pytest
from typing import List, Dict, Any

# Import modules to test
from semantic_clustering.clustering import (
    cluster_keywords,
    optimize_clusters,
    extract_cluster_labels,
    evaluate_clusters,
    get_keywords_embeddings_matrix
)

# Import test utilities
from tests import create_test_sample_data, TEST_CONFIG

# Optional: Use hypothesis for property-based testing
from hypothesis import given, strategies as st

class TestEmbeddingsGeneration:
    """
    Test suite for keyword embeddings generation.
    """
    
    def test_embeddings_matrix_generation(self):
        """
        Test generation of embeddings matrix from keywords.
        """
        # Create sample keywords
        keywords = [
            "digital marketing strategy",
            "seo best practices",
            "content marketing tips",
            "social media advertising"
        ]
        
        # Generate embeddings
        embeddings_matrix, processed_keywords = get_keywords_embeddings_matrix(keywords)
        
        # Assertions
        assert embeddings_matrix is not None
        assert len(processed_keywords) > 0
        assert embeddings_matrix.shape[0] == len(processed_keywords)
        assert embeddings_matrix.shape[1] > 0  # Embedding dimensionality
    
    def test_embeddings_with_different_models(self):
        """
        Test embeddings generation with different models.
        """
        keywords = ["machine learning", "artificial intelligence"]
        models = [
            "all-MiniLM-L6-v2",
            "paraphrase-multilingual-MiniLM-L12-v2"
        ]
        
        for model in models:
            embeddings_matrix, processed_keywords = get_keywords_embeddings_matrix(
                keywords, 
                model_name=model
            )
            
            assert embeddings_matrix is not None
            assert len(processed_keywords) > 0
    
    def test_embeddings_empty_input(self):
        """
        Test embeddings generation with empty input.
        """
        embeddings_matrix, processed_keywords = get_keywords_embeddings_matrix([])
        
        assert len(embeddings_matrix) == 0
        assert len(processed_keywords) == 0

class TestClusteringAlgorithms:
    """
    Test suite for clustering algorithms.
    """
    
    @pytest.mark.parametrize("method", [
        "kmeans", 
        "dbscan", 
        "hdbscan", 
        "agglomerative"
    ])
    def test_cluster_keywords(self, method: str):
        """
        Test keyword clustering with different algorithms.
        
        Args:
            method: Clustering method to test
        """
        # Create sample keywords
        keywords = [
            "digital marketing strategy",
            "seo best practices",
            "content marketing tips",
            "social media advertising",
            "email marketing automation",
            "search engine optimization",
            "content creation strategies",
            "social media management"
        ]
        
        # Clustering parameters
        params = {
            "kmeans": {"n_clusters": 3},
            "agglomerative": {"n_clusters": 3}
        }
        
        # Get clustering parameters for the method
        method_params = params.get(method, {})
        
        # Perform clustering
        clusters, embeddings_matrix, processed_keywords, cluster_model = cluster_keywords(
            keywords,
            method=method,
            **method_params
        )
        
        # Assertions
        assert isinstance(clusters, dict)
        assert len(clusters) > 0
        assert embeddings_matrix is not None
        assert len(processed_keywords) > 0
        
        # Check cluster contents
        total_keywords = sum(len(cluster_keywords) for cluster_keywords in clusters.values())
        assert total_keywords == len(processed_keywords)
    
    def test_cluster_optimization(self):
        """
        Test cluster count optimization.
        """
        keywords = [
            "digital marketing strategy",
            "seo best practices",
            "content marketing tips",
            "social media advertising",
            "email marketing automation",
            "search engine optimization",
            "content creation strategies",
            "social media management"
        ]
        
        # Optimize clusters
        optimal_clusters, best_score, scores = optimize_clusters(
            keywords,
            min_clusters=2,
            max_clusters=5,
            method="kmeans"
        )
        
        # Assertions
        assert isinstance(optimal_clusters, int)
        assert 2 <= optimal_clusters <= 5
        assert isinstance(best_score, float)
        assert best_score >= 0
        assert len(scores) > 0

class TestClusterLabeling:
    """
    Test suite for cluster labeling methods.
    """
    
    @pytest.mark.parametrize("method", [
        "frequent", 
        "tfidf", 
        "centroid"
    ])
    def test_extract_cluster_labels(self, method: str):
        """
        Test cluster label extraction methods.
        
        Args:
            method: Label extraction method
        """
        # Sample clusters
        clusters = {
            "0": ["digital marketing strategy", "seo marketing"],
            "1": ["content creation tips", "content marketing guide"],
            "2": ["social media advertising", "social media management"]
        }
        
        # Extract labels
        labels = extract_cluster_labels(clusters, method=method)
        
        # Assertions
        assert isinstance(labels, dict)
        assert len(labels) == len(clusters)
        for cluster_id, label in labels.items():
            assert isinstance(label, str)
            assert len(label) > 0

class TestClusterEvaluation:
    """
    Test suite for cluster evaluation metrics.
    """
    
    def test_cluster_evaluation(self):
        """
        Test comprehensive cluster evaluation.
        """
        # Sample keywords and embeddings
        keywords = [
            "digital marketing strategy",
            "seo best practices",
            "content marketing tips",
            "social media advertising",
            "email marketing automation"
        ]
        
        # Generate embeddings
        embeddings_matrix, processed_keywords = get_keywords_embeddings_matrix(keywords)
        
        # Perform clustering
        clusters, _, _, cluster_model = cluster_keywords(
            keywords, 
            method="kmeans", 
            n_clusters=2
        )
        
        # Extract labels
        labels = cluster_model.labels_
        
        # Evaluate clusters
        metrics = evaluate_clusters(clusters, embeddings_matrix, labels)
        
        # Assertions
        expected_metrics = [
            "total_clusters", 
            "total_items", 
            "min_size", 
            "max_size", 
            "mean_size", 
            "median_size",
            "silhouette_score", 
            "calinski_harabasz_score", 
            "davies_bouldin_score"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
        
        # Check numeric constraints
        assert metrics['total_clusters'] > 0
        assert metrics['total_items'] > 0
        assert 0 <= metrics['silhouette_score'] <= 1

# Integration fixture
@pytest.fixture
def sample_keywords_data():
    """
    Fixture to provide sample keywords for integrated testing.
    
    Returns:
        List of sample keywords
    """
    sample_path = create_test_sample_data('keywords')
    
    with open(sample_path, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        keywords = [line.split(',')[0].strip() for line in f]
    
    return keywords

def test_clustering_pipeline_integration(sample_keywords_data):
    """
    Integration test for complete clustering pipeline.
    
    Args:
        sample_keywords_data: Fixture with sample keywords
    """
    # Generate embeddings
    embeddings_matrix, processed_keywords = get_keywords_embeddings_matrix(
        sample_keywords_data
    )
    
    # Perform clustering
    clusters, _, _, cluster_model = cluster_keywords(
        sample_keywords_data, 
        method="kmeans",
        n_clusters=3
    )
    
    # Extract cluster labels
    labels = extract_cluster_labels(clusters)
    
    # Evaluate clusters
    metrics = evaluate_clusters(clusters, embeddings_matrix, cluster_model.labels_)
    
    # Assertions
    assert len(clusters) > 0
    assert len(labels) > 0
    assert 'silhouette_score' in metrics

# Performance testing
def test_clustering_performance(benchmark, sample_keywords_data):
    """
    Performance benchmark for clustering pipeline.
    
    Args:
        benchmark: Pytest-benchmark fixture
        sample_keywords_data: Fixture with sample keywords
    """
    def clustering_workflow():
        embeddings_matrix, _ = get_keywords_embeddings_matrix(sample_keywords_data)
        return cluster_keywords(
            sample_keywords_data, 
            method="kmeans",
            n_clusters=3
        )
    
    result = benchmark(clustering_workflow)
    
    # Basic result validation
    clusters, _, _, _ = result
    assert len(clusters) > 0
