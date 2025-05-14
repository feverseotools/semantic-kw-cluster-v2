#!/usr/bin/env python3
"""
Language Model Installation Script for Semantic Keyword Clustering.

This script provides a command-line interface to install and manage 
language models for NLP preprocessing and clustering.
"""

import argparse
import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Centralized language model configurations
LANGUAGE_MODELS: Dict[str, Dict[str, str]] = {
    # Core models (easy to install)
    "en": {
        "name": "en_core_web_sm",
        "language": "English",
        "description": "Small English language model"
    },
    "es": {
        "name": "es_core_news_sm",
        "language": "Spanish", 
        "description": "Small Spanish language model"
    },
    "fr": {
        "name": "fr_core_news_sm",
        "language": "French",
        "description": "Small French language model"
    },
    "de": {
        "name": "de_core_news_sm", 
        "language": "German",
        "description": "Small German language model"
    },
    "pl": {
        "name": "pl_core_news_sm",
        "language": "Polish", 
        "description": "Small Polish language model"
    },
    # Additional models (less common)
    "pt": {
        "name": "pt_core_news_sm",
        "language": "Portuguese",
        "description": "Small Portuguese language model"
    },
    "it": {
        "name": "it_core_news_sm",
        "language": "Italian", 
        "description": "Small Italian language model"
    },
    "nl": {
        "name": "nl_core_news_sm",
        "language": "Dutch",
        "description": "Small Dutch language model"
    }
}

def check_python_version() -> bool:
    """
    Verify Python version compatibility.
    
    Returns:
        bool: True if Python version is compatible, False otherwise
    """
    min_version = (3, 8)
    current_version = sys.version_info
    
    if current_version < min_version:
        logger.error(
            f"Unsupported Python version. "
            f"Required: {'.'.join(map(str, min_version))}+, "
            f"Current: {'.'.join(map(str, current_version[:3]))}"
        )
        return False
    return True

def check_spacy_installation() -> bool:
    """
    Check if spaCy is installed.
    
    Returns:
        bool: True if spaCy is installed, False otherwise
    """
    try:
        import spacy
        logger.info(f"SpaCy version: {spacy.__version__}")
        return True
    except ImportError:
        logger.warning("SpaCy is not installed. Please install it first.")
        return False

def list_available_models() -> None:
    """
    List all available language models.
    """
    logger.info("Available Language Models:")
    for code, model_info in LANGUAGE_MODELS.items():
        print(f"  [{code}] {model_info['language']}: {model_info['name']} - {model_info['description']}")

def download_model(model_code: str) -> bool:
    """
    Download a specific language model.
    
    Args:
        model_code: Two-letter language code
        
    Returns:
        bool: True if model installation was successful, False otherwise
    """
    if model_code not in LANGUAGE_MODELS:
        logger.error(f"Invalid language code: {model_code}")
        return False
    
    model_name = LANGUAGE_MODELS[model_code]['name']
    
    try:
        logger.info(f"Attempting to download {model_name}...")
        
        # Use subprocess to call spacy download
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"Successfully downloaded {model_name}")
            return True
        else:
            logger.error(f"Error downloading {model_name}:")
            logger.error(result.stderr)
            return False
    
    except Exception as e:
        logger.error(f"Installation error for {model_name}: {e}")
        return False

def download_nltk_resources() -> bool:
    """
    Download essential NLTK resources.
    
    Returns:
        bool: True if resources were downloaded successfully, False otherwise
    """
    try:
        import nltk
        
        # List of NLTK resources to download
        resources = [
            'stopwords',
            'punkt',
            'wordnet',
            'omw-1.4'
        ]
        
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
                logger.info(f"Downloaded NLTK resource: {resource}")
            except Exception as res_error:
                logger.warning(f"Failed to download {resource}: {res_error}")
        
        return True
    
    except ImportError:
        logger.error("NLTK is not installed. Please install it first.")
        return False

def main():
    """
    Main CLI entry point for model installation script.
    """
    # Verify Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Language Model Installation Tool for Semantic Keyword Clustering"
    )
    
    # Add mutually exclusive argument group
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-l", "--list", 
        action="store_true", 
        help="List available language models"
    )
    group.add_argument(
        "-d", "--download", 
        type=str, 
        metavar="LANG_CODE",
        help="Download a specific language model (e.g., 'en', 'es')"
    )
    group.add_argument(
        "-a", "--all", 
        action="store_true", 
        help="Download all available language models"
    )
    
    parser.add_argument(
        "-n", "--nltk", 
        action="store_true", 
        help="Download NLTK resources"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check spaCy installation
    if not check_spacy_installation():
        logger.error("Please install spaCy first: pip install spacy")
        sys.exit(1)
    
    # Execute based on arguments
    if args.list:
        list_available_models()
    
    elif args.download:
        if download_model(args.download):
            sys.exit(0)
        else:
            sys.exit(1)
    
    elif args.all:
        success_count = sum(download_model(code) for code in LANGUAGE_MODELS.keys())
        if success_count == len(LANGUAGE_MODELS):
            logger.info("All models downloaded successfully!")
            sys.exit(0)
        else:
            logger.warning(f"{success_count}/{len(LANGUAGE_MODELS)} models downloaded")
            sys.exit(1)
    
    # Download NLTK resources if requested
    if args.nltk:
        if download_nltk_resources():
            sys.exit(0)
        else:
            sys.exit(1)
    
    # If no arguments provided, show help
    parser.print_help()
    sys.exit(1)

if __name__ == "__main__":
    main()
