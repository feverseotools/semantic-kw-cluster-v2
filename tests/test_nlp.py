"""
NLP Module Test Suite for Semantic Keyword Clustering.

This module contains unit tests for natural language processing 
functionalities including preprocessing, intent classification, 
and language model management.
"""

import os
import pytest
from typing import List, Dict, Any

# Import modules to test
from semantic_clustering.nlp import (
    preprocess_text,
    preprocess_keywords,
    classify_search_intent,
    load_spacy_model_by_language
)

# Import test utilities
from tests import create_test_sample_data, TEST_CONFIG

# Optional: Use hypothesis for property-based testing
from hypothesis import given, strategies as st

class TestNLPPreprocessing:
    """
    Test suite for text preprocessing functionality.
    """
    
    @pytest.mark.parametrize("input_text, expected", [
        ("Machine Learning", "machine learning"),
        ("Data Science!", "data science"),
        ("Python 3.8 is Great!", "python is great"),
        ("   Whitespace   Test   ", "whitespace test"),
    ])
    def test_basic_preprocessing(self, input_text: str, expected: str):
        """
        Test basic text preprocessing functionality.
        
        Args:
            input_text: Input text to preprocess
            expected: Expected preprocessed output
        """
        result = preprocess_text(input_text)
        assert result == expected, f"Failed to preprocess: {input_text}"
    
    def test_preprocessing_empty_input(self):
        """
        Test preprocessing with empty or None input.
        """
        assert preprocess_text("") == ""
        assert preprocess_text(None) == ""
    
    @pytest.mark.parametrize("language", ["english", "spanish", "french"])
    def test_language_specific_preprocessing(self, language: str):
        """
        Test preprocessing with different language settings.
        
        Args:
            language: Language code for preprocessing
        """
        text = "This is a test sentence with some stopwords"
        result = preprocess_text(text, language=language)
        
        # Basic checks
        assert isinstance(result, str)
        assert len(result) > 0
        assert "is" not in result.split()  # Common stopword
    
    @given(st.text())
    def test_preprocessing_robustness(self, random_text: str):
        """
        Property-based testing for preprocessing robustness.
        
        Args:
            random_text: Randomly generated text
        """
        try:
            result = preprocess_text(random_text)
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"Preprocessing failed for input: {random_text}")

class TestKeywordPreprocessing:
    """
    Test suite for keyword preprocessing functionality.
    """
    
    def test_keyword_preprocessing(self):
        """
        Test preprocessing a list of keywords.
        """
        keywords = [
            "Digital Marketing Strategy",
            "SEO Best Practices",
            "Machine Learning Algorithms"
        ]
        
        processed_keywords = preprocess_keywords(keywords)
        
        assert len(processed_keywords) == len(keywords)
        for keyword in processed_keywords:
            assert isinstance(keyword, str)
            assert len(keyword) > 0
    
    def test_empty_keyword_list(self):
        """
        Test preprocessing an empty list of keywords.
        """
        assert preprocess_keywords([]) == []

class TestSearchIntentClassification:
    """
    Test suite for search intent classification.
    """
    
    @pytest.mark.parametrize("keywords, expected_primary_intent", [
        (["how to learn python"], "Informational"),
        (["buy python course"], "Transactional"),
        (["best python tutorial"], "Commercial"),
        (["python official website"], "Navigational")
    ])
    def test_intent_classification(self, keywords: List[str], expected_primary_intent: str):
        """
        Test search intent classification for various keyword types.
        
        Args:
            keywords: List of keywords to classify
            expected_primary_intent: Expected primary intent
        """
        result = classify_search_intent(keywords)
        
        assert "primary_intent" in result
        assert "scores" in result
        assert "evidence" in result
        
        # Validate primary intent
        assert result["primary_intent"] == expected_primary_intent
        
        # Check scores are percentages
        for intent, score in result["scores"].items():
            assert 0 <= score <= 100
    
    def test_empty_keywords_intent(self):
        """
        Test intent classification with empty keywords.
        """
        result = classify_search_intent([])
        
        assert result["primary_intent"] == "Unknown"
        assert all(0 <= score <= 100 for score in result["scores"].values())

class TestLanguageModelManagement:
    """
    Test suite for language model management.
    """
    
    @pytest.mark.parametrize("language", [
        "English", "Spanish", "French", "German", "Polish"
    ])
    def test_spacy_model_loading(self, language: str):
        """
        Test loading spaCy models for different languages.
        
        Args:
            language: Language name to load model for
        """
        model = load_spacy_model_by_language(language)
        
        if language in ["English", "Spanish", "French", "German", "Polish"]:
            # For core languages, model should load
            assert model is not None, f"Failed to load {language} model"
        
            # Basic model capabilities
            sample_text = "This is a test sentence."
            doc = model(sample_text)
            
            assert len(list(doc.sents)) > 0
            assert len(list(doc)) > 0
    
    def test_unsupported_language(self):
        """
        Test loading an unsupported language model.
        """
        model = load_spacy_model_by_language("Klingon")
        assert model is None

# Fixture for integration testing
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

def test_nlp_pipeline_integration(sample_keywords_data):
    """
    Integration test for NLP pipeline.
    
    Args:
        sample_keywords_data: Fixture with sample keywords
    """
    # Preprocess keywords
    processed_keywords = preprocess_keywords(sample_keywords_data)
    
    # Classify intents
    intent_results = classify_search_intent(processed_keywords)
    
    # Assertions
    assert len(processed_keywords) > 0
    assert "primary_intent" in intent_results
    assert "scores" in intent_results

# Optional: Performance testing
def test_nlp_performance(benchmark):
    """
    Performance benchmark for keyword preprocessing.
    
    Args:
        benchmark: Pytest-benchmark fixture
    """
    keywords = [
        "Digital Marketing Strategy" * 10,
        "SEO Best Practices" * 10,
        "Machine Learning Algorithms" * 10
    ]
    
    result = benchmark(preprocess_keywords, keywords)
    
    assert len(result) == len(keywords)
