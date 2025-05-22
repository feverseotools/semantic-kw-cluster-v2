import os

# Populate library availability and language models before UI import
from utils import SPACY_LANGUAGE_MODELS, LIBRARIES

def detect_libraries():
    """
    Detect availability of key libraries and populate the LIBRARIES dict.
    """
    try:
        import spacy  # noqa: F401
        LIBRARIES['spacy_base_available'] = True
    except ImportError:
        LIBRARIES['spacy_base_available'] = False

    try:
        from textblob import TextBlob  # noqa: F401
        LIBRARIES['textblob_available'] = True
    except ImportError:
        LIBRARIES['textblob_available'] = False

    try:
        import sentence_transformers  # noqa: F401
        LIBRARIES['sentence_transformers_available'] = True
    except ImportError:
        LIBRARIES['sentence_transformers_available'] = False

    try:
        import openai  # noqa: F401
        LIBRARIES['openai_available'] = True
    except ImportError:
        LIBRARIES['openai_available'] = False

# Define default spaCy language models
SPACY_LANGUAGE_MODELS.update({
    'en': 'en_core_web_sm',
    'es': 'es_core_news_sm',
    'de': 'de_core_news_sm',
    'fr': 'fr_core_news_sm',
})

# Run library detection
detect_libraries()

# Now import the UI
import streamlit as st
from ui import main

if __name__ == '__main__':
    # Entry point for Streamlit
    main()
