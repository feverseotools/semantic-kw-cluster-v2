"""
Language model management for Semantic Keyword Clustering.

This module handles the loading and configuration of language models
for different languages.
"""

import logging
import streamlit as st

# Configure logging
logger = logging.getLogger(__name__)

# Check if spaCy is available
try:
    import spacy
    spacy_base_available = True
except ImportError:
    spacy_base_available = False
    logger.warning("spaCy not available. Install with pip install spacy")

# Mapping for known spaCy language models (if installed).
# If these models are not installed, spaCy loading will fail and fallback to other methods.
SPACY_LANGUAGE_MODELS = {
    # Core models that come with standard spaCy installations
    "English": {"model": "en_core_web_sm", "fallback": True},
    "Spanish": {"model": "es_core_news_sm", "fallback": True},
    "French": {"model": "fr_core_news_sm", "fallback": True},
    "German": {"model": "de_core_news_sm", "fallback": True},
    "Italian": {"model": "it_core_news_sm", "fallback": True},
    "Portuguese": {"model": "pt_core_news_sm", "fallback": True},
    "Dutch": {"model": "nl_core_news_sm", "fallback": True},
    "Polish": {"model": "pl_core_news_sm", "fallback": True},
    
    # Models that need separate installation
    "Swedish": {"model": "sv_core_news_sm", "fallback": False, "install_cmd": "python -m spacy download sv_core_news_sm"},
    "Norwegian": {"model": "nb_core_news_sm", "fallback": False, "install_cmd": "python -m spacy download nb_core_news_sm"},
    "Danish": {"model": "da_core_news_sm", "fallback": False, "install_cmd": "python -m spacy download da_core_news_sm"},
    "Greek": {"model": "el_core_news_sm", "fallback": False, "install_cmd": "python -m spacy download el_core_news_sm"},
    "Romanian": {"model": "ro_core_news_sm", "fallback": False, "install_cmd": "python -m spacy download ro_core_news_sm"},
    
    # Languages with specific community models
    "Japanese": {"model": "ja_core_news_sm", "fallback": False, "install_cmd": "python -m spacy download ja_core_news_sm"},
    "Korean": {"model": "ko_core_news_sm", "fallback": False, "install_cmd": "python -m spacy download ko_core_news_sm"},
    
    # Languages without specific models - using multi-language fallback
    "Icelandic": {"model": None, "fallback": True, "alt_model": "xx_ent_wiki_sm"},
    "Lithuanian": {"model": None, "fallback": True, "alt_model": "xx_ent_wiki_sm"},
    "Brazilian Portuguese": {"model": "pt_core_news_sm", "fallback": True}  # Same as Portuguese
}

def get_embedding_model(model_name=None, device=None):
    """
    Returns a sentence embedding model for generating embeddings.
    
    Args:
        model_name (str, optional): The name of the sentence-transformers model to use.
                                   Default is 'all-MiniLM-L6-v2' if None.
        device (str, optional): Device to use ('cpu' or 'cuda'). If None, autodetects.
    
    Returns:
        SentenceTransformer: A model that can transform text into embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        if model_name is None:
            model_name = "all-MiniLM-L6-v2"  # Default lightweight model
        
        if device is None:
            # Auto-detect device
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
                logger.warning("PyTorch not available, using CPU for embeddings")
        
        logger.info(f"Loading embedding model: {model_name} on {device}")
        model = SentenceTransformer(model_name, device=device)
        return model
        
    except ImportError:
        error_msg = "sentence-transformers not installed. Install with: pip install sentence-transformers"
        logger.error(error_msg)
        if st:  # Check if we're in a Streamlit context
            st.error(error_msg)
        raise ImportError(error_msg)
    except Exception as e:
        error_msg = f"Error loading embedding model: {str(e)}"
        logger.error(error_msg)
        if st:  # Check if we're in a Streamlit context
            st.error(error_msg)
        raise

def load_spacy_model_by_language(selected_language):
    """
    Try to load a spaCy model for the given language.
    
    Args:
        selected_language (str): The language name to load a model for
        
    Returns:
        spacy.Language or None: The loaded spaCy model, or None if it couldn't be loaded
    """
    if not spacy_base_available:
        logger.warning("spaCy is not installed. Install it with: pip install spacy")
        if st:  # Check if we're in a Streamlit context
            st.warning("spaCy is not installed. Some advanced NLP features will be unavailable.")
        return None

    # Get language config
    lang_config = SPACY_LANGUAGE_MODELS.get(selected_language)
    if lang_config is None:
        message = f"No spaCy configuration for {selected_language}. Using fallback processing."
        logger.warning(message)
        if st:  # Check if we're in a Streamlit context
            st.warning(message)
        return None
    
    model_name = lang_config.get("model")
    allows_fallback = lang_config.get("fallback", False)
    alt_model = lang_config.get("alt_model")
    install_cmd = lang_config.get("install_cmd", f"python -m spacy download {model_name}" if model_name else "")
    
    # Case 1: Language has a specific model
    if model_name:
        try:
            model = spacy.load(model_name)
            message = f"Loaded spaCy model for {selected_language}: {model_name}"
            logger.info(message)
            if st:  # Check if we're in a Streamlit context
                st.success(f"✅ {message}")
            return model
        except OSError:
            # Model not installed
            if allows_fallback:
                message = f"spaCy model for {selected_language} not installed. Using fallback processing."
                if install_cmd:
                    message += f" To improve results, install: {install_cmd}"
                logger.warning(message)
                if st:  # Check if we're in a Streamlit context
                    st.warning(message)
                return None
            else:
                message = f"spaCy model for {selected_language} is required but not installed. Install with: {install_cmd}"
                logger.error(message)
                if st:  # Check if we're in a Streamlit context
                    st.error(message)
                return None
        except Exception as e:
            if allows_fallback:
                message = f"Error loading spaCy model: {str(e)}. Using fallback processing."
                logger.warning(message)
                if st:  # Check if we're in a Streamlit context
                    st.warning(message)
                return None
            else:
                message = f"Error loading spaCy model: {str(e)}"
                logger.error(message)
                if st:  # Check if we're in a Streamlit context
                    st.error(message)
                return None
    
    # Case 2: Language uses alternative model
    elif alt_model:
        try:
            model = spacy.load(alt_model)
            message = f"Using alternative spaCy model for {selected_language}: {alt_model}"
            logger.info(message)
            if st:  # Check if we're in a Streamlit context
                st.info(message)
            return model
        except Exception as e:
            message = f"Error loading alternative model: {str(e)}. Using fallback processing."
            logger.warning(message)
            if st:  # Check if we're in a Streamlit context
                st.warning(message)
            return None
    
    # Case 3: No model available
    else:
        message = f"No spaCy model available for {selected_language}. Using fallback processing."
        logger.warning(message)
        if st:  # Check if we're in a Streamlit context
            st.warning(message)
        return None

def check_available_models():
    """
    Check which spaCy language models are installed.
    
    Returns:
        dict: Dictionary of language names and whether they're installed
    """
    if not spacy_base_available:
        return {"status": "spaCy not installed"}
    
    results = {}
    for language, config in SPACY_LANGUAGE_MODELS.items():
        model_name = config.get("model")
        if not model_name and config.get("alt_model"):
            model_name = config.get("alt_model")
        
        if model_name:
            try:
                spacy.load(model_name)
                results[language] = "Installed"
            except OSError:
                results[language] = "Not installed"
            except Exception as e:
                results[language] = f"Error: {str(e)}"
        else:
            results[language] = "No model available"
    
    return results
