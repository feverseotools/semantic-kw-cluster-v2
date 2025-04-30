import os
import sys
import time
import json
import re
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import nltk
import plotly.express as px
import plotly.graph_objects as go

# System and Library Checks
def safe_import(library_name):
    """
    Safely import libraries with error tracking
    """
    try:
        __import__(library_name)
        return True
    except ImportError:
        st.error(f"Failed to import {library_name}")
        st.error(traceback.format_exc())
        return False

def check_system_info():
    """
    Display system and library version information
    """
    st.sidebar.header("🖥 System Information")
    st.sidebar.write(f"Python Version: {sys.version}")
    st.sidebar.write(f"Streamlit Version: {st.__version__}")

    # Library version checks
    libraries = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('nltk', 'NLTK'),
        ('sklearn', 'Scikit-Learn'),
        ('plotly', 'Plotly'),
        ('openai', 'OpenAI'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('spacy', 'spaCy'),
        ('textblob', 'TextBlob')
    ]

    for lib, name in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'Unknown')
            st.sidebar.write(f"{name} Version: {version}")
        except ImportError:
            st.sidebar.error(f"{name} not installed")

# NLTK Resource Download
@st.cache_resource
def download_nltk_resources():
    """
    Download necessary NLTK resources
    """
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        return True
    except Exception as e:
        st.error(f"NLTK resource download failed: {e}")
        return False

# Import Utility Modules
def import_utility_modules():
    """
    Import and check utility modules
    """
    utility_modules = [
        'utils.preprocessing',
        'utils.embedding',
        'utils.clustering',
        'utils.intent',
        'utils.naming',
        'utils.visualization'
    ]

    for module in utility_modules:
        try:
            __import__(module)
        except ImportError:
            st.error(f"Failed to import {module}")
            st.error(traceback.format_exc())

# Early Initialization
download_nltk_resources()
import_utility_modules()

# Page Configuration
st.set_page_config(
    page_title="Semantic Keyword Clustering",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Application Function
def main():
    # Display System Information
    check_system_info()

    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
            color: #2c3e50;
        }
    </style>
    """, unsafe_allow_html=True)

    # Application Title
    st.markdown("<div class='main-header'>🔍 Semantic Keyword Clustering</div>", unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    This advanced tool clusters keywords semantically, helping you understand 
    search intent, customer journey, and content strategy.
    
    **Key Features:**
    - Semantic clustering using NLP
    - Search intent classification
    - Customer journey mapping
    - OpenAI-powered insights (optional)
    """)

    # Sidebar Configuration
    with st.sidebar:
        st.header("Clustering Configuration")
        
        # File Upload
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        # Clustering Parameters
        num_clusters = st.slider("Number of Clusters", 2, 50, 10)
        
        # Language Selection
        languages = ["English", "Spanish", "French", "German", "Portuguese"]
        language = st.selectbox("Select Language", languages)
        
        # OpenAI Integration
        st.subheader("OpenAI Integration (Optional)")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        
        # Start Clustering Button
        start_clustering = st.button("Start Clustering", use_container_width=True)

    # Clustering Process
    if start_clustering and uploaded_file:
        try:
            # Import modules dynamically to handle potential import errors
            from utils.preprocessing import preprocess_keywords
            from utils.embedding import generate_embeddings
            from utils.clustering import cluster_keywords
            from utils.intent import classify_search_intent
            from utils.naming import generate_cluster_names
            
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Preprocessing
            with st.spinner("Preprocessing Keywords..."):
                processed_keywords = preprocess_keywords(df['keyword'], language)
            
            # Generate Embeddings
            with st.spinner("Generating Semantic Embeddings..."):
                embeddings = generate_embeddings(
                    processed_keywords, 
                    openai_api_key=openai_api_key
                )
            
            # Clustering
            with st.spinner("Performing Semantic Clustering..."):
                cluster_results = cluster_keywords(
                    embeddings, 
                    num_clusters=num_clusters
                )
            
            # Display Results
            st.success(f"Created {num_clusters} semantic clusters!")
            
            # Visualization and Analysis to be added...
        
        except Exception as e:
            st.error(f"Clustering failed: {e}")
            st.error(traceback.format_exc())

    # Sample Data Download
    st.sidebar.markdown("### Sample Data")
    if st.sidebar.button("Download Sample CSV"):
        # Implement sample CSV generation logic
        pass

# Application Entry Point
if __name__ == "__main__":
    main()
