import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import json
import re
import nltk
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Semantic Keyword Clustering",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; margin-bottom: 1rem;}
    .sub-header {font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem;}
    .info-box {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
    .success-box {background-color: #d4edda; color: #155724; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
    .warning-box {background-color: #fff3cd; color: #856404; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
    .error-box {background-color: #f8d7da; color: #721c24; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
    .highlight {background-color: #fffbcc; padding: 0.2rem 0.5rem; border-radius: 0.2rem;}
    .intent-box {padding: 8px; border-radius: 5px; margin-bottom: 10px;}
    .intent-info {background-color: #e3f2fd; border-left: 5px solid #2196f3;}
    .intent-nav {background-color: #e8f5e9; border-left: 5px solid #4caf50;}
    .intent-trans {background-color: #fff3e0; border-left: 5px solid #ff9800;}
    .intent-comm {background-color: #f3e5f5; border-left: 5px solid #9c27b0;}
    .intent-mixed {background-color: #f5f5f5; border-left: 5px solid #9e9e9e;}
    .tag {display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; margin-right: 5px;}
    .info-tag {background-color: #e3f2fd; color: #0d47a1;}
    .commercial-tag {background-color: #f3e5f5; color: #4a148c;}
    .transactional-tag {background-color: #fff3e0; color: #e65100;}
    .navigational-tag {background-color: #e8f5e9; color: #1b5e20;}
    .journey-box {padding: 10px; margin-bottom: 10px; border-radius: 5px;}
    .journey-early {background-color: #e8f5e9; border-left: 5px solid #43a047;}
    .journey-middle {background-color: #e3f2fd; border-left: 5px solid #1e88e5;}
    .journey-late {background-color: #fff3e0; border-left: 5px solid #ff9800;}
</style>
""", unsafe_allow_html=True)
# Download NLTK resources at startup
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        return True
    except Exception as e:
        st.error(f"Failed to download NLTK resources: {str(e)}")
        return False

nltk_download_successful = download_nltk_resources()

# Initialize session state variables
if 'process_complete' not in st.session_state:
    st.session_state['process_complete'] = False
if 'clustering_results' not in st.session_state:
    st.session_state['clustering_results'] = None
if 'cluster_names' not in st.session_state:
    st.session_state['cluster_names'] = {}
if 'cluster_intents' not in st.session_state:
    st.session_state['cluster_intents'] = {}

# Available language options
LANGUAGE_OPTIONS = {
    "English": "en",
    "Spanish": "es",
    "German": "de",
    "French": "fr",
    "Portuguese": "pt",
    "Italian": "it"
}

# Function to generate a sample CSV
def generate_sample_csv(with_header=True):
    if with_header:
        header = "Keyword,search_volume,competition,cpc,month01,month02,month03,month04,month05,month06,month07,month08,month09,month10,month11,month12\n"
        data = [
            "running shoes,5400,0.75,1.25,450,460,470,480,490,500,510,520,530,540,550,560",
            "nike shoes,8900,0.82,1.78,700,720,740,760,780,800,820,840,860,880,900,920",
            "adidas sneakers,3200,0.65,1.12,260,270,280,290,300,310,320,330,340,350,360,370",
            "hiking boots,2800,0.45,0.89,230,240,250,260,270,280,290,300,310,320,330,340",
            "women's running shoes,4100,0.68,1.35,340,350,360,370,380,390,400,410,420,430,440,450",
            "best running shoes 2023,3100,0.78,1.52,280,290,300,310,320,330,340,350,360,370,380,390",
            "how to choose running shoes,2500,0.42,0.95,220,230,240,250,260,270,280,290,300,310,320,330",
            "running shoes for flat feet,1900,0.56,1.28,170,180,190,200,210,220,230,240,250,260,270,280",
            "trail running shoes reviews,1700,0.64,1.42,150,160,170,180,190,200,210,220,230,240,250,260",
            "buy nike air zoom,1500,0.87,1.95,130,140,150,160,170,180,190,200,210,220,230,240"
        ]
        return header + "\n".join(data)
    else:
        # No header, just keywords
        return "\n".join([
            "running shoes",
            "nike shoes",
            "adidas sneakers",
            "hiking boots",
            "women's running shoes",
            "best running shoes 2023",
            "how to choose running shoes",
            "running shoes for flat feet",
            "trail running shoes reviews",
            "buy nike air zoom"
        ])

# Function to calculate estimated API cost
def calculate_api_cost(num_keywords, model="gpt-3.5-turbo", num_clusters=10):
    # Updated pricing (approximations)
    EMBEDDING_COST_PER_1K = 0.0001  # text-embedding-ada-002
    GPT35_COST_PER_1K = 0.002  # gpt-3.5-turbo input tokens
    GPT4_COST_PER_1K = 0.03  # gpt-4 input tokens
    
    # Estimate tokens
    avg_tokens_per_keyword = 5
    estimated_embedding_tokens = num_keywords * avg_tokens_per_keyword
    embedding_cost = (estimated_embedding_tokens / 1000) * EMBEDDING_COST_PER_1K
    
    # Clustering naming cost
    avg_tokens_per_cluster_prompt = 200
    if model == "gpt-3.5-turbo":
        naming_cost = (avg_tokens_per_cluster_prompt * num_clusters / 1000) * GPT35_COST_PER_1K
    else:
        naming_cost = (avg_tokens_per_cluster_prompt * num_clusters / 1000) * GPT4_COST_PER_1K
    
    total_cost = embedding_cost + naming_cost
    
    return {
        "embedding_cost": embedding_cost,
        "naming_cost": naming_cost,
        "total_cost": total_cost
    }
