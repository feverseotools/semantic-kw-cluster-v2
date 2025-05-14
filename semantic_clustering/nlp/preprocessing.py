"""
Text preprocessing module for Semantic Keyword Clustering.

This module contains functions for cleaning and preprocessing text
before embedding and clustering.
"""

import re
import logging
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Configure logging
logger = logging.getLogger(__name__)

# Check for TextBlob availability
try:
    from textblob import TextBlob
    textblob_available = True
except ImportError:
    textblob_available = False
    logger.info("TextBlob not available. Install with: pip install textblob")

# Try to download NLTK resources at import time
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    logger.warning(f"Error downloading NLTK resources: {str(e)}")
    logger.warning("Some preprocessing features may not work correctly.")

def preprocess_text(text, use_lemmatization=True, language="english"):
    """
    Basic text preprocessing using NLTK.
    
    Args:
        text (str): The text to preprocess
        use_lemmatization (bool): Whether to apply lemmatization
        language (str): Language code for stopwords (e.g., "english", "spanish")
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        text = text.lower()
        
        # Language-adapted tokenization
        try:
            tokens = word_tokenize(text, language=language)
        except:
            tokens = word_tokenize(text)  # Fallback to default tokenizer
            logger.warning(f"Using default tokenizer for language: {language}")
        
        # Language-specific stopwords handling
        try:
            stop_words = set(stopwords.words(language))
        except:
            # Default stopwords if NLTK fails
            stop_words = {'a','an','the','and','or','but','if','because','as','what','in','on','to','for'}
            logger.warning(f"Using default stopwords for language: {language}")
            
            # Polish-specific stopwords if that language is selected
            if language.lower() in ["polish", "pl", "polski"]:
                polish_stopwords = {
                    'i', 'w', 'na', 'z', 'do', 'jest', 'to', 'nie', 'się', 'że', 'dla', 'a', 'od', 'jak',
                    'po', 'co', 'tak', 'za', 'przez', 'o', 'lub', 'ale', 'już', 'być', 'ten', 'który',
                    'też', 'tylko', 'jeszcze', 'przy', 'może', 'ich', 'bardzo', 'tam', 'jako', 'wszystko',
                    'gdy', 'więc', 'zawsze', 'aby', 'kiedy', 'można', 'jeśli', 'bez', 'gdzie', 'czy'
                }
                stop_words.update(polish_stopwords)
        
        # Filter tokens that are words and not stopwords
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        
        # Lemmatization according to language
        if use_lemmatization:
            if language.lower() in ["polish", "pl", "polski"]:
                # For Polish, use a simplified approach
                # Remove common Polish endings (simplified)
                simplified_tokens = []
                for t in tokens:
                    if len(t) > 4:
                        for suffix in ['ów', 'ami', 'ach', 'om', 'owi', 'em', 'ie', 'ego', 'emu', 'ymi', 'ich', 'imi']:
                            if t.endswith(suffix) and len(t) - len(suffix) > 3:
                                t = t[:-len(suffix)]
                                break
                    simplified_tokens.append(t)
                tokens = simplified_tokens
            else:
                # For other languages, use WordNetLemmatizer
                try:
                    lemmatizer = WordNetLemmatizer()
                    tokens = [lemmatizer.lemmatize(t) for t in tokens]
                except Exception as lemma_error:
                    logger.warning(f"Lemmatization failed: {str(lemma_error)}")
        
        return " ".join(tokens)
    except Exception as e:
        logger.error(f"Error in preprocess_text: {str(e)}")
        return text.lower() if isinstance(text, str) else ""

def enhanced_preprocessing(text, use_lemmatization, spacy_nlp=None, language="english"):
    """
    Enhanced preprocessing using spaCy or fallback with TextBlob.
    
    Args:
        text (str): The text to preprocess
        use_lemmatization (bool): Whether to apply lemmatization
        spacy_nlp: Loaded spaCy model
        language (str): Language code for preprocessing
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        if spacy_nlp is not None:  # We use the loaded spaCy model
            doc = spacy_nlp(text.lower())
            entities = [ent.text for ent in doc.ents]
            tokens = []
            for token in doc:
                if not token.is_stop and token.is_alpha and len(token.text) > 1:
                    tokens.append(token.lemma_)
            
            # Bigrams
            bigrams = []
            for i in range(len(doc) - 1):
                if (not doc[i].is_stop and not doc[i+1].is_stop
                    and doc[i].is_alpha and doc[i+1].is_alpha):
                    bigrams.append(f"{doc[i].lemma_}_{doc[i+1].lemma_}")
            
            processed_parts = tokens + bigrams + entities
            return " ".join(processed_parts)
        
        elif textblob_available:
            from textblob import TextBlob
            blob = TextBlob(text.lower())
            noun_phrases = list(blob.noun_phrases)
            
            # Get stopwords according to language
            try:
                stop_words = set(stopwords.words(language))
            except:
                stop_words = {'a','an','the','and','or','but','if','because','as','what','in','on','to','for'}
                # Add Polish stopwords if needed
                if language.lower() in ["polish", "pl", "polski"]:
                    polish_stopwords = {
                        'i', 'w', 'na', 'z', 'do', 'jest', 'to', 'nie', 'się', 'że', 'dla', 'a', 'od', 'jak',
                        'po', 'co', 'tak', 'za', 'przez', 'o', 'lub', 'ale', 'już', 'być', 'ten', 'który',
                        'też', 'tylko', 'jeszcze', 'przy', 'może', 'ich', 'bardzo', 'tam', 'jako', 'wszystko',
                        'gdy', 'więc', 'zawsze', 'aby', 'kiedy', 'można', 'jeśli', 'bez', 'gdzie', 'czy'
                    }
                    stop_words.update(polish_stopwords)
            
            words = [w for w in blob.words if len(w) > 1 and w.lower() not in stop_words]
            
            if use_lemmatization:
                if language.lower() in ["polish", "pl", "polski"]:
                    # Simplify Polish words (basic approach)
                    lemmas = []
                    for w in words:
                        if len(w) > 4:
                            for suffix in ['ów', 'ami', 'ach', 'om', 'owi', 'em', 'ie', 'ego', 'emu', 'ymi', 'ich', 'imi']:
                                if w.endswith(suffix) and len(w) - len(suffix) > 3:
                                    w = w[:-len(suffix)]
                                    break
                        lemmas.append(w)
                else:
                    # For other languages use standard lemmatizer
                    lemmatizer = WordNetLemmatizer()
                    lemmas = [lemmatizer.lemmatize(w) for w in words]
                
                processed_parts = lemmas + noun_phrases
            else:
                processed_parts = words + noun_phrases
            
            return " ".join(processed_parts)
        
        else:
            # fallback to standard NLTK
            logger.info("Using fallback preprocessing with NLTK")
            return preprocess_text(text, use_lemmatization, language)
    
    except Exception as e:
        logger.error(f"Error in enhanced_preprocessing: {str(e)}")
        return text.lower() if isinstance(text, str) else ""

def preprocess_keywords(keywords, use_advanced=True, spacy_nlp=None, selected_language="English"):
    """
    Preprocess a list of keywords with progress tracking.
    
    Args:
        keywords (list): List of keywords to process
        use_advanced (bool): Whether to use advanced preprocessing with spaCy/TextBlob
        spacy_nlp: Loaded spaCy model
        selected_language (str): Language name for preprocessing
        
    Returns:
        list: List of preprocessed keywords
    """
    processed_keywords = []
    total = len(keywords)
    
    # Map the selected language to NLTK codes
    language_map = {
        "English": "english",
        "Spanish": "spanish",
        "French": "french",
        "German": "german",
        "Italian": "italian",
        "Portuguese": "portuguese",
        "Polish": "polish",
        # Add more mappings as needed
    }
    nltk_language = language_map.get(selected_language, "english")
    
    # Log preprocessing approach
    if use_advanced:
        if spacy_nlp is not None:
            logger.info(f"Using advanced preprocessing with spaCy for {selected_language}")
            if st:  # Check if in Streamlit context
                st.success(f"Using advanced preprocessing with spaCy for {selected_language}")
        elif textblob_available:
            logger.info(f"Using fallback preprocessing with TextBlob for {selected_language}")
            if st:  # Check if in Streamlit context
                st.success(f"Using fallback preprocessing with TextBlob for {selected_language}")
        else:
            logger.info(f"Using standard preprocessing with NLTK for {selected_language}")
            if st:  # Check if in Streamlit context
                st.info(f"Using standard preprocessing with NLTK for {selected_language}")
    else:
        logger.info(f"Using standard preprocessing with NLTK for {selected_language} (advanced preprocessing disabled)")
        if st:  # Check if in Streamlit context
            st.info(f"Using standard preprocessing with NLTK for {selected_language} (advanced preprocessing disabled)")
    
    # Initialize progress tracking in Streamlit if available
    progress_bar = None
    if st:  # Check if we're in a Streamlit context
        progress_bar = st.progress(0)
    
    # Process each keyword
    for i, keyword in enumerate(keywords):
        if use_advanced and (spacy_nlp is not None or textblob_available):
            processed_keywords.append(enhanced_preprocessing(keyword, True, spacy_nlp, nltk_language))
        else:
            processed_keywords.append(preprocess_text(keyword, True, nltk_language))
        
        # Update progress
        if progress_bar and i % 10 == 0:  # Update every 10 items
            progress_bar.progress(min(i / total, 1.0))
    
    # Complete progress
    if progress_bar:
        progress_bar.progress(1.0)
    
    return processed_keywords
