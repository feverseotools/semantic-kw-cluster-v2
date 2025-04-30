import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import streamlit as st

# Language ISO code to NLTK language name mapping
LANGUAGE_NLTK_MAP = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese"
}

@st.cache_data
def preprocess_keywords(keywords, language="en"):
    """
    Preprocess a list of keywords for semantic analysis.
    
    Args:
        keywords (list): List of keywords to preprocess
        language (str): Language ISO code (en, es, fr, de, pt, it)
        
    Returns:
        list: List of preprocessed keywords
    """
    # Ensure NLTK resources are downloaded
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    # Get language-specific stopwords
    nltk_lang = LANGUAGE_NLTK_MAP.get(language, "english")
    try:
        stop_words = set(stopwords.words(nltk_lang))
    except:
        # Fallback to English if language not available
        stop_words = set(stopwords.words('english'))
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Process each keyword
    processed_keywords = []
    
    for keyword in keywords:
        try:
            # Convert to lowercase and remove special characters
            if not isinstance(keyword, str):
                keyword = str(keyword)
            
            keyword = keyword.lower()
            keyword = re.sub(r'[^\w\s]', ' ', keyword)
            
            # Tokenize
            tokens = word_tokenize(keyword)
            
            # Remove stopwords and lemmatize
            filtered_tokens = []
            for token in tokens:
                if token.isalpha() and token not in stop_words:
                    if language == "en":  # Only lemmatize English for now
                        token = lemmatizer.lemmatize(token)
                    filtered_tokens.append(token)
            
            # Join tokens back into a string
            processed_keyword = " ".join(filtered_tokens)
            processed_keywords.append(processed_keyword)
        
        except Exception as e:
            # In case of error, keep the original keyword
            st.warning(f"Error preprocessing keyword '{keyword}': {str(e)}")
            processed_keywords.append(keyword if isinstance(keyword, str) else "")
    
    return processed_keywords
