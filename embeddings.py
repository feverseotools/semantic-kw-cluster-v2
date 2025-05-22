import os
import time
import numpy as np
import streamlit as st
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Constants for OpenAI integration (can be overridden via environment variables)
OPENAI_TIMEOUT = float(os.getenv('OPENAI_TIMEOUT', '60.0'))
OPENAI_MAX_RETRIES = int(os.getenv('OPENAI_MAX_RETRIES', '3'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '100'))

@st.cache_resource(ttl=3600)
def load_sentence_transformer(model_name: str):
    """
    Load a SentenceTransformer model and cache it for reuse.

    Args:
        model_name (str): Name of the pretrained SentenceTransformer model.

    Returns:
        SentenceTransformer instance or None if unavailable.
    """
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except ImportError:
        logger.warning("sentence-transformers library is not installed.")
        return None
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer model '{model_name}': {e}")
        return None


def create_openai_client(api_key: str):
    """
    Create an OpenAI client with timeout and retry settings.

    Args:
        api_key (str): Your OpenAI API key.

    Returns:
        OpenAI client instance or None if unavailable.
    """
    try:
        import openai
        openai.api_key = api_key
        openai.timeout = OPENAI_TIMEOUT
        openai.max_retries = OPENAI_MAX_RETRIES
        return openai
    except ImportError:
        logger.warning("openai library is not installed.")
        return None
    except Exception as e:
        logger.error(f"Failed to create OpenAI client: {e}")
        return None


def generate_openai_embeddings(text_list, client, model: str, batch_size: int = BATCH_SIZE):
    """
    Generate embeddings for a list of texts via OpenAI API, in batches.

    Args:
        text_list (List[str]): Texts to embed.
        client: OpenAI client instance.
        model (str): Embedding model name, e.g., 'text-embedding-3-small'.
        batch_size (int): Number of texts per API call.

    Returns:
        np.ndarray: 2D array of embeddings.
    """
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        try:
            response = client.Embedding.create(model=model, input=batch)
            for data in response['data']:
                embeddings.append(data['embedding'])
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    return np.array(embeddings)


def generate_embeddings_with_fallback(text_list, openai_api_key: str = None,
                                     openai_model: str = 'text-embedding-3-small',
                                     st_model_name: str = 'all-MiniLM-L6-v2'):
    """
    Generate embeddings using OpenAI, falling back to SentenceTransformer or TF-IDF.

    Args:
        text_list (List[str]): Texts to embed.
        openai_api_key (str): API key for OpenAI.
        openai_model (str): OpenAI embedding model name.
        st_model_name (str): SentenceTransformer model name.

    Returns:
        np.ndarray: Array of embeddings.
    """
    # Attempt OpenAI
    if openai_api_key:
        client = create_openai_client(openai_api_key)
        if client:
            try:
                return generate_openai_embeddings(text_list, client, openai_model)
            except Exception:
                st.warning("OpenAI embedding failed. Falling back to local models.")

    # Attempt SentenceTransformer
    st_model = load_sentence_transformer(st_model_name)
    if st_model:
        try:
            return st_model.encode(text_list, show_progress_bar=True)
        except Exception as e:
            logger.error(f"SentenceTransformer encoding error: {e}")

    # Fallback to TF-IDF
    logger.info("Using TF-IDF as a fallback embedding method.")
    vectorizer = TfidfVectorizer(max_features=300)
    tfidf_matrix = vectorizer.fit_transform([t or "" for t in text_list])
    return tfidf_matrix.toarray()


def generate_embeddings(dataframe, text_column: str = 'text',
                       openai_api_key: str = None,
                       openai_model: str = 'text-embedding-3-small',
                       st_model_name: str = 'all-MiniLM-L6-v2'):
    """
    Convenience wrapper to generate embeddings for a DataFrame column.

    Args:
        dataframe (pandas.DataFrame): DataFrame containing the text.
        text_column (str): Column name with text data.
        openai_api_key (str): API key for OpenAI.
        openai_model (str): OpenAI embedding model.
        st_model_name (str): SentenceTransformer model.

    Returns:
        np.ndarray: Embedding array.
    """
    texts = dataframe[text_column].fillna("").tolist()
    return generate_embeddings_with_fallback(
        texts,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        st_model_name=st_model_name
    )
