"""
Export Module Test Suite for Semantic Keyword Clustering.

This module contains comprehensive unit and integration tests
for various export formats and functionality.
"""

import os
import pytest
import tempfile
from typing import Dict, List, Any

# Import modules to test
from semantic_clustering.export import (
    export_to_json,
    export_to_excel,
    export_to_html,
    export_to_pdf,
    export_jsonl_for_elasticsearch
)

# Import test utilities
from tests import create_test_sample_data, TEST_CONFIG, cleanup_test_data

# Optional: Use hypothesis for property-based testing
from hypothesis import given, strategies as st

class TestJSONExport:
    """
    Test suite for JSON export functionality.
    """
    
    def setup_method(self):
        """
        Set up test environment before each test method.
        """
        self.sample_clusters = {
            "0": ["digital marketing strategy", "seo marketing"],
            "1": ["content creation tips", "content marketing guide"],
            "2": ["social media advertising", "social media management"]
        }
        
        self.cluster_labels = {
            "0": "Digital Marketing",
            "1": "Content Strategy",
            "2": "Social Media"
        }
        
        self.evaluation_metrics = {
            "total_clusters": 3,
            "silhouette_score": 0.75,
            "total_keywords": 6
        }
    
    def test_json_export_basic(self):
        """
        Test basic JSON export functionality.
        """
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            try:
                # Export to JSON
                result = export_to_json(
                    self.sample_clusters,
                    temp_file.name,
                    cluster_labels=self.cluster_labels,
                    evaluation_metrics=self.evaluation_metrics
                )
                
                # Assertions
                assert result is True
                assert os.path.exists(temp_file.name)
                assert os.path.getsize(temp_file.name) > 0
            
            finally:
                # Clean up temporary file
                os.unlink(temp_file.name)
    
    def test_json_export_no_metadata(self):
        """
        Test JSON export without metadata.
        """
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            try:
                # Export to JSON without metadata
                result = export_to_json(
                    self.sample_clusters,
                    temp_file.name,
                    include_metadata=False
                )
                
                # Assertions
                assert result is True
                assert os.path.exists(temp_file.name)
                assert os.path.getsize(temp_file.name) > 0
            
            finally:
                # Clean up temporary file
                os.unlink(temp_file.name)

class TestExcelExport:
    """
    Test suite for Excel export functionality.
    """
    
    def setup_method(self):
        """
        Set up test environment before each test method.
        """
        self.sample_clusters = {
            "0": ["digital marketing strategy", "seo marketing"],
            "1": ["content creation tips", "content marketing guide"],
            "2": ["social media advertising", "social media management"]
        }
        
        self.cluster_labels = {
            "0": "Digital Marketing",
            "1": "Content Strategy",
            "2": "Social Media"
        }
        
        self.evaluation_metrics = {
            "total_clusters": 3,
            "silhouette_score": 0.75,
            "total_keywords": 6
        }
    
    def test_excel_export(self):
        """
        Test basic Excel export functionality.
        """
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            try:
                # Export to Excel
                result = export_to_excel(
                    self.sample_clusters,
                    temp_file.name,
                    cluster_labels=self.cluster_labels,
                    evaluation_metrics=self.evaluation_metrics
                )
                
                # Assertions
                assert result is True
                assert os.path.exists(temp_file.name)
                assert os.path.getsize(temp_file.name) > 0
            
            finally:
                # Clean up temporary file
                os.unlink(temp_file.name)
    
    def test_excel_export_no_metrics(self):
        """
        Test Excel export without metrics.
        """
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            try:
                # Export to Excel without metrics
                result = export_to_excel(
                    self.sample_clusters,
                    temp_file.name,
                    include_metrics=False
                )
                
                # Assertions
                assert result is True
                assert os.path.exists(temp_file.name)
                assert os.path.getsize(temp_file.name) > 0
            
            finally:
                # Clean up temporary file
                os.unlink(temp_file.name)

class TestHTMLExport:
    """
    Test suite for HTML export functionality.
    """
    
    def setup_method(self):
        """
        Set up test environment before each test method.
        """
        self.sample_clusters = {
            "0": ["digital marketing strategy", "seo marketing"],
            "1": ["content creation tips", "content marketing guide"],
            "2": ["social media advertising", "social media management"]
        }
        
        self.cluster_labels = {
            "0": "Digital Marketing",
            "1": "Content Strategy",
            "2": "Social Media"
        }
        
        self.evaluation_metrics = {
            "total_clusters": 3,
            "silhouette_score": 0.75,
            "total_keywords": 6
        }
        
        # Simulated 2D embeddings for visualization
        import numpy as np
        self.embeddings_2d = np.random.rand(6, 2)
        self.labels = np.array([0, 0, 1, 1, 2, 2])
    
    def test_html_export(self):
        """
        Test basic HTML export functionality.
        """
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp_file:
            try:
                # Export to HTML
                result = export_to_html(
                    self.sample_clusters,
                    temp_file.name,
                    cluster_labels=self.cluster_labels,
                    evaluation_metrics=self.evaluation_metrics,
                    embeddings_2d=self.embeddings_2d,
                    labels=self.labels
                )
                
                # Assertions
                assert result is True
                assert os.path.exists(temp_file.name)
                assert os.path.getsize(temp_file.name) > 0
                
                # Optional: Read file content to do basic validation
                with open(temp_file.name, 'r', encoding='utf-8') as f:
                    content = f.read()
                    assert "Semantic Keyword Clustering" in content
                    assert all(label in content for label in self.cluster_labels.values())
            
            finally:
                # Clean up temporary file
                os.unlink(temp_file.name)

class TestPDFExport:
    """
    Test suite for PDF export functionality.
    """
    
    def setup_method(self):
        """
        Set up test environment before each test method.
        """
        self.sample_clusters = {
            "0": ["digital marketing strategy", "seo marketing"],
            "1": ["content creation tips", "content marketing guide"],
            "2": ["social media advertising", "social media management"]
        }
        
        self.cluster_labels = {
            "0": "Digital Marketing",
            "1": "Content Strategy",
            "2": "Social Media"
        }
        
        self.evaluation_metrics = {
            "total_clusters": 3,
            "silhouette_score": 0.75,
            "total_keywords": 6
        }
        
        # Simulated 2D embeddings for visualization
        import numpy as np
        self.embeddings_2d = np.random.rand(6, 2)
        self.labels = np.array([0, 0, 1, 1, 2, 2])
    
    def test_pdf_export(self):
        """
        Test basic PDF export functionality.
        """
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            try:
                # Export to PDF
                result = export_to_pdf(
                    self.sample_clusters,
                    temp_file.name,
                    cluster_labels=self.cluster_labels,
                    evaluation_metrics=self.evaluation_metrics,
                    embeddings_2d=self.embeddings_2d,
                    labels=self.labels
                )
                
                # Assertions
                assert result is True
                assert os.path.exists(temp_file.name)
                assert os.path.getsize(temp_file.name) > 0
            
            finally:
                # Clean up temporary file
                os.unlink(temp_file.name)

class TestElasticsearchExport:
    """
    Test suite for Elasticsearch export functionality.
    """
    
    def setup_method(self):
        """
        Set up test environment before each test method.
        """
        self.sample_clusters = {
            "0": ["digital marketing strategy", "seo marketing"],
            "1": ["content creation tips", "content marketing guide"],
            "2": ["social media advertising", "social media management"]
        }
        
        self.cluster_labels = {
            "0": "Digital Marketing",
            "1": "Content Strategy",
            "2": "Social Media"
        }
    
    def test_elasticsearch_export(self):
        """
        Test Elasticsearch export functionality.
        """
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as temp_file:
            try:
                # Export for Elasticsearch
                result = export_jsonl_for_elasticsearch(
                    self.sample_clusters,
                    temp_file.name,
                    cluster_labels=self.cluster_labels
                )
                
                # Assertions
                assert result is True
                assert os.path.exists(temp_file.name)
                assert os.path.getsize(temp_file.name) > 0
                
                # Optional: Read file and validate content
                with open(temp_file.name, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Each keyword should have two lines (action and document)
                    assert len(lines) == len(sum(self.sample_clusters.values(), [])) * 2
            
            finally:
                # Clean up temporary file
                os.unlink(temp_file.name)

# Integration Fixture
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

def test_export_pipeline_integration(sample_keywords_data):
    """
    Integration test for export pipeline.
    
    Args:
        sample_keywords_data: Fixture with sample keywords
    """
    # Simulate clustering results
    clusters = {
        "0": sample_keywords_data[:len(sample_keywords_data)//2],
        "1": sample_keywords_data[len(sample_keywords_data)//2:]
    }
    
    # Export formats to test
    export_formats = [
        (export_to_json, '.json'),
        (export_to_excel, '.xlsx'),
        (export_to_html, '.html'),
        (export_to_pdf, '.pdf')
    ]
    
    # Test each export format
    for export_func, file_ext in export_formats:
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
            try:
                # Export using function
                result = export_func(
                    clusters,
                    temp_file.name,
                    cluster_labels={k: f"Cluster {k}" for k in clusters.keys()}
                )
                
                # Assertions
                assert result is True
                assert os.path.exists(temp_file.name)
                assert os.path.getsize(temp_file.name) > 0
            
            finally:
                # Clean up temporary file
                os.unlink(temp_file.name)

# Cleanup after tests
def test_cleanup():
    """
    Cleanup test data after test suite completion.
    """
    cleanup_test_data()
