"""
Natural Language Processing module for Semantic Keyword Clustering.

This module handles text preprocessing, language model management,
and search intent classification.
"""

# Import key functions for easier access from the nlp module
from semantic_clustering.nlp.preprocessing import (
    preprocess_text,
    preprocess_keywords,
    enhanced_preprocessing
)
from semantic_clustering.nlp.intent import (
    classify_search_intent,
    analyze_cluster_for_intent_flow,
    extract_features_for_intent
)
from semantic_clustering.nlp.models import (
    load_spacy_model_by_language,
    SPACY_LANGUAGE_MODELS
)

# Define what's available when using `from semantic_clustering.nlp import *`
__all__ = [
    "preprocess_text",
    "preprocess_keywords",
    "enhanced_preprocessing",
    "classify_search_intent",
    "analyze_cluster_for_intent_flow",
    "extract_features_for_intent",
    "load_spacy_model_by_language",
    "SPACY_LANGUAGE_MODELS"
]
