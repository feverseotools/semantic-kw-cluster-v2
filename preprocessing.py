import re
import logging
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

@st.cache_resource
def download_nltk_resources():
    """
    Download necessary NLTK resources: stopwords, punkt, wordnet, omw-1.4.
    Returns True if the download succeeds, False otherwise.
    """
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        logger.info("NLTK resources downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {e}")
        return False

@st.cache_resource(ttl=7200)
def load_spacy_model_by_language(selected_language, model_map, base_available):
    """
    Load a spaCy model for the specified language if spaCy is available and the model exists in model_map.

    Args:
        selected_language (str): Language code (e.g., 'en', 'es').
        model_map (dict): Mapping of language codes to spaCy model names.
        base_available (bool): Indicates if spaCy base installation is present.

    Returns:
        A spaCy Language object or None if loading fails.
    """
    if not base_available:
        logger.warning("spaCy is not available")
        return None

    model_name = model_map.get(selected_language)
    if model_name is None:
        logger.info(f"No spaCy model configured for language '{selected_language}'")
        return None

    try:
        import spacy
        return spacy.load(model_name)
    except Exception as e:
        logger.warning(f"Failed to load spaCy model '{model_name}': {e}")
        return None


def preprocess_text(text, use_lemmatization=True):
    """
    Basic text preprocessing:
      - Lowercase conversion.
      - Tokenization and stopword removal.
      - Alphabetic token filtering.
      - (Optional) Lemmatization.

    Args:
        text (str): The input text to preprocess.
        use_lemmatization (bool): Whether to perform lemmatization.

    Returns:
        str: The preprocessed text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    try:
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        if use_lemmatization:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
    except Exception:
        # Fallback to simple split if NLTK fails
        tokens = re.sub(r"[^a-z ]", "", text).split()

    return " ".join(tokens)


def enhanced_preprocessing(text, use_lemmatization, spacy_nlp, textblob_available=False):
    """
    Advanced preprocessing:
      - If spacy_nlp is provided, use spaCy for tokenization, lemmatization, bigram and entity extraction.
      - If TextBlob is available, use blob.words and blob.noun_phrases.
      - Otherwise, fallback to basic preprocess_text.

    Args:
        text (str): Input text.
        use_lemmatization (bool): Whether to lemmatize tokens.
        spacy_nlp: spaCy Language model instance or None.
        textblob_available (bool): Whether TextBlob is available.

    Returns:
        str: The processed text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    try:
        if spacy_nlp:
            doc = spacy_nlp(text.lower())
            tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
            bigrams = [
                f"{doc[i].lemma_}_{doc[i+1].lemma_}"
                for i in range(len(doc) - 1)
                if doc[i].is_alpha and doc[i+1].is_alpha and not doc[i].is_stop and not doc[i+1].is_stop
            ]
            entities = [ent.text for ent in doc.ents]
            return " ".join(tokens + bigrams + entities)

        if textblob_available:
            from textblob import TextBlob
            blob = TextBlob(text.lower())
            words = [w for w in blob.words if len(w) > 1]
            if use_lemmatization:
                lemmatizer = WordNetLemmatizer()
                words = [lemmatizer.lemmatize(w) for w in words]
            return " ".join(words + list(blob.noun_phrases))

        # Fallback to basic preprocessing
        return preprocess_text(text, use_lemmatization)

    except Exception as e:
        logger.warning(f"Enhanced preprocessing failed: {e}")
        return preprocess_text(text, use_lemmatization)


def preprocess_keywords(keywords, use_advanced, spacy_nlp=None, textblob_available=False):
    """
    Preprocess a list of keywords with a Streamlit progress bar.

    Args:
        keywords (List[str]): List of keywords to process.
        use_advanced (bool): Whether to apply enhanced preprocessing.
        spacy_nlp: spaCy Language model instance or None.
        textblob_available (bool): Whether TextBlob is available.

    Returns:
        List[str]: List of processed keywords.
    """
    processed = []
    total = len(keywords)
    progress_bar = st.progress(0, text="Preprocessing keywords...")

    for idx, keyword in enumerate(keywords):
        if use_advanced:
            processed.append(enhanced_preprocessing(keyword, True, spacy_nlp, textblob_available))
        else:
            processed.append(preprocess_text(keyword, True))

        # Update progress every 10% or at the end
        if (idx + 1) % max(1, total // 10) == 0 or idx == total - 1:
            progress_bar.progress((idx + 1) / total)

    return processed
preprocessing.py
