"""
Semantic Keyword Clustering.

A package for clustering keywords based on semantic similarity,
analyzing search intent, and mapping to customer journey stages.
"""

__version__ = "0.1.0"
__author__ = "Max Sanchez"
__email__ = "maximo.sanchez@feverup.com"

# Import main components for easier access
from semantic_clustering.nlp.preprocessing import preprocess_text, preprocess_keywords
from semantic_clustering.nlp.intent import classify_search_intent
from semantic_clustering.clustering.algorithms import improved_clustering, refine_clusters
from semantic_clustering.clustering.embeddings import generate_embeddings

# Define what's available when using `from semantic_clustering import *`
__all__ = [
    "preprocess_text",
    "preprocess_keywords",
    "classify_search_intent",
    "improved_clustering",
    "refine_clusters",
    "generate_embeddings",
]

# Add version info function
def get_version():
    """Return the package version as a string."""
    return __version__
