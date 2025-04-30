import numpy as np
import streamlit as st

@st.cache_data
def generate_embeddings(preprocessed_keywords, openai_api_key=None, language="en"):
    """
    Generate embeddings for preprocessed keywords using OpenAI or SentenceTransformers.
    
    Args:
        preprocessed_keywords (list): List of preprocessed keywords
        openai_api_key (str, optional): OpenAI API key for using their embeddings
        language (str): Language ISO code (en, es, fr, de, pt, it)
        
    Returns:
        numpy.ndarray: Array of embeddings
    """
    if not preprocessed_keywords:
        st.error("No keywords to generate embeddings for")
        return np.array([])
    
    # Remove empty strings
    preprocessed_keywords = [k for k in preprocessed_keywords if k.strip()]
    
    if len(preprocessed_keywords) == 0:
        st.error("All keywords are empty after preprocessing")
        return np.array([])
    
    # If OpenAI API key is provided, use their embeddings
    if openai_api_key:
        try:
            return generate_openai_embeddings(preprocessed_keywords, openai_api_key)
        except Exception as e:
            st.error(f"Error generating OpenAI embeddings: {str(e)}")
            st.warning("Falling back to SentenceTransformers")
    
    # Otherwise, use SentenceTransformers
    return generate_sentence_transformer_embeddings(preprocessed_keywords, language)

@st.cache_data
def generate_openai_embeddings(preprocessed_keywords, api_key):
    """Generate embeddings using OpenAI's API."""
    try:
        import openai
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        # Generate embeddings in batches to avoid rate limits
        embeddings = []
        batch_size = 100  # OpenAI can handle up to 2048 items per request, but we'll be conservative
        
        with st.spinner(f"Generating OpenAI embeddings for {len(preprocessed_keywords)} keywords..."):
            for i in range(0, len(preprocessed_keywords), batch_size):
                batch = preprocessed_keywords[i:i+batch_size]
                
                response = client.embeddings.create(
                    input=batch,
                    model="text-embedding-ada-002"
                )
                
                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Update progress
                progress = min(1.0, (i + len(batch)) / len(preprocessed_keywords))
                st.progress(progress)
        
        st.success("✅ OpenAI embeddings generated successfully")
        return np.array(embeddings)
    
    except Exception as e:
        st.error(f"Failed to generate OpenAI embeddings: {str(e)}")
        raise

@st.cache_data
def generate_sentence_transformer_embeddings(preprocessed_keywords, language="en"):
    """Generate embeddings using SentenceTransformers."""
    try:
        from sentence_transformers import SentenceTransformer
        
        # Select model based on language
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"  # Good multilingual model
        
        with st.spinner(f"Loading SentenceTransformer model '{model_name}'..."):
            model = SentenceTransformer(model_name)
        
        with st.spinner(f"Generating embeddings for {len(preprocessed_keywords)} keywords..."):
            # Generate embeddings in batches
            embeddings = []
            batch_size = 256
            
            for i in range(0, len(preprocessed_keywords), batch_size):
                batch = preprocessed_keywords[i:i+batch_size]
                batch_embeddings = model.encode(batch, show_progress_bar=False)
                embeddings.extend(batch_embeddings)
                
                # Update progress
                progress = min(1.0, (i + len(batch)) / len(preprocessed_keywords))
                st.progress(progress)
        
        st.success("✅ SentenceTransformer embeddings generated successfully")
        return np.array(embeddings)
    
    except Exception as e:
        st.error(f"Failed to generate SentenceTransformer embeddings: {str(e)}")
        
        # Last resort: TF-IDF vectors
        st.warning("Falling back to TF-IDF vectorization")
        return generate_tfidf_embeddings(preprocessed_keywords)

@st.cache_data
def generate_tfidf_embeddings(preprocessed_keywords):
    """Generate TF-IDF vectors as a last resort."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        with st.spinner("Generating TF-IDF vectors..."):
            vectorizer = TfidfVectorizer(max_features=300)
            tfidf_matrix = vectorizer.fit_transform(preprocessed_keywords)
            
        st.success("✅ TF-IDF vectors generated successfully")
        return tfidf_matrix.toarray()
    
    except Exception as e:
        st.error(f"Failed to generate TF-IDF vectors: {str(e)}")
        
        # Ultimate fallback: random vectors (not ideal but prevents complete failure)
        st.warning("Using random vectors as ultimate fallback")
        return np.random.rand(len(preprocessed_keywords), 100)
