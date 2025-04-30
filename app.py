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
# Main application function
def main():
    # Header
    st.markdown("<div class='main-header'>🔍 Semantic Keyword Clustering</div>", unsafe_allow_html=True)
    st.markdown("""
    This app clusters your keywords based on semantic similarity, helping you organize them for SEO and paid campaigns.
    Upload a CSV with your keywords and discover meaningful groups to target more effectively.
    """)

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV with keywords", type=["csv"])
        
        # Sample data options
        if uploaded_file is None:
            st.markdown("### Sample Data")
            use_sample = st.checkbox("Use sample data", value=False)
            sample_type = st.radio(
                "Sample type:", 
                ["With header (like Keyword Planner)", "Without header (just keywords)"],
                index=0
            )
        
        # CSV format
        st.markdown("### CSV Format")
        csv_format = st.radio(
            "Select CSV format",
            ["With header (column names in first row)", "No header (just keywords)"],
            index=0
        )
        
        # Language selection
        st.markdown("### Language")
        selected_language_name = st.selectbox(
            "Select language of keywords",
            list(LANGUAGE_OPTIONS.keys()),
            index=0
        )
        selected_language = LANGUAGE_OPTIONS[selected_language_name]
        
        # OpenAI API Key
        st.markdown("### OpenAI API (Optional)")
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="If not provided, free SentenceTransformers will be used instead"
        )
        
        # Clustering parameters
        st.markdown("### Clustering Parameters")
        num_clusters = st.slider(
            "Number of clusters",
            min_value=2,
            max_value=50,
            value=10,
            help="How many keyword groups to create"
        )
        
        # Advanced parameters
        with st.expander("Advanced Parameters", expanded=False):
            pca_variance = st.slider(
                "PCA explained variance (%)",
                min_value=60,
                max_value=99,
                value=95,
                help="Higher keeps more information but higher dimensionality"
            )
            
            openai_model = st.selectbox(
                "OpenAI model for cluster naming",
                ["gpt-3.5-turbo", "gpt-4"],
                index=0,
                help="GPT-4 gives better names but costs more"
            )
        
        # Cost estimate
        if openai_api_key:
            with st.expander("💰 Cost Estimate"):
                if uploaded_file:
                    try:
                        df_temp = pd.read_csv(uploaded_file)
                        num_keywords_estimate = len(df_temp)
                    except:
                        num_keywords_estimate = 100
                else:
                    num_keywords_estimate = 10
                
                cost = calculate_api_cost(num_keywords_estimate, openai_model, num_clusters)
                
                st.markdown(f"""
                **Estimated costs for {num_keywords_estimate} keywords:**
                - Embedding: ${cost['embedding_cost']:.5f}
                - Cluster naming: ${cost['naming_cost']:.5f}
                - **Total: ${cost['total_cost']:.5f}**
                """)
                
                st.info("These are estimates. Actual cost may vary.")
# Main content area
    if not st.session_state['process_complete']:
        # Download sample data button
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Start by uploading your keyword data")
            st.markdown("""
            Upload a CSV file with your keywords, or use the sample data to test the app.
            """)
        with col2:
            with st.expander("Need a sample file?"):
                st.markdown("Download a sample CSV to see the expected format:")
                
                if st.button("Download with headers (Keyword Planner format)"):
                    csv_data = generate_sample_csv(with_header=True)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="sample_keywords_with_header.csv",
                        mime="text/csv"
                    )
                
                if st.button("Download without headers (just keywords)"):
                    csv_data = generate_sample_csv(with_header=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="sample_keywords_no_header.csv",
                        mime="text/csv"
                    )
        
        # Process data
        process_col1, process_col2, process_col3 = st.columns([1, 2, 1])
        with process_col2:
            if st.button("Start Semantic Clustering", type="primary", use_container_width=True):
                try:
                    # Load data
                    df = None
                    if uploaded_file is not None:
                        if csv_format == "With header (column names in first row)":
                            df = pd.read_csv(uploaded_file)
                            if "Keyword" in df.columns:
                                df.rename(columns={"Keyword": "keyword"}, inplace=True)
                            if "keyword" not in df.columns:
                                st.error("CSV must have a 'keyword' or 'Keyword' column.")
                                return
                        else:
                            df = pd.read_csv(uploaded_file, header=None, names=["keyword"])
                    elif use_sample:
                        if sample_type == "With header (like Keyword Planner)":
                            sample_data = StringIO(generate_sample_csv(with_header=True))
                            df = pd.read_csv(sample_data)
                        else:
                            sample_data = StringIO(generate_sample_csv(with_header=False))
                            df = pd.read_csv(sample_data, header=None, names=["keyword"])
                    
                    if df is None or len(df) == 0:
                        st.error("No data to process. Please upload a CSV or use sample data.")
                        return
                    
                    # Display dataset info
                    st.success(f"Loaded {len(df)} keywords")
                    
                    # Preprocessing
                    with st.spinner("Preprocessing keywords..."):
                        from utils.preprocessing import preprocess_keywords
                        df['processed_keyword'] = preprocess_keywords(
                            df['keyword'].tolist(), 
                            language=selected_language
                        )
                    
                    # Generate embeddings
                    with st.spinner("Generating semantic embeddings..."):
                        from utils.embedding import generate_embeddings
                        embeddings = generate_embeddings(
                            df['processed_keyword'].tolist(),
                            openai_api_key=openai_api_key,
                            language=selected_language
                        )
                    
                    # Clustering
                    with st.spinner("Clustering keywords..."):
                        from utils.clustering import cluster_keywords
                        cluster_results = cluster_keywords(
                            embeddings, 
                            num_clusters=num_clusters,
                            pca_variance=pca_variance/100
                        )
                        
                        df['cluster_id'] = cluster_results['labels']
                        
                        # Get representative keywords for each cluster
                        representative_keywords = {}
                        for cluster_id in range(1, num_clusters+1):
                            cluster_mask = df['cluster_id'] == cluster_id
                            if any(cluster_mask):
                                cluster_keywords = df.loc[cluster_mask, 'keyword'].tolist()
                                representative_keywords[cluster_id] = cluster_keywords[:min(10, len(cluster_keywords))]
# Generate cluster names using OpenAI if API key is provided
                    cluster_names = {}
                    if openai_api_key:
                        with st.spinner("Generating descriptive cluster names..."):
                            try:
                                from utils.naming import generate_cluster_names
                                cluster_names = generate_cluster_names(
                                    representative_keywords, 
                                    openai_api_key, 
                                    model=openai_model
                                )
                            except Exception as e:
                                st.error(f"Error generating cluster names: {str(e)}")
                                # Fallback to generic names
                                for cluster_id in representative_keywords.keys():
                                    cluster_names[cluster_id] = (f"Cluster {cluster_id}", f"Group of related keywords")
                    else:
                        # Generate simple names based on top keywords
                        for cluster_id, keywords in representative_keywords.items():
                            if keywords:
                                name = f"Cluster {cluster_id}: {keywords[0]}"
                                desc = f"Keywords related to {', '.join(keywords[:3])}"
                                cluster_names[cluster_id] = (name, desc)
                    
                    # Apply names to dataframe
                    df['cluster_name'] = df['cluster_id'].map(lambda x: cluster_names.get(x, (f"Cluster {x}", ""))[0])
                    df['cluster_description'] = df['cluster_id'].map(lambda x: cluster_names.get(x, ("", f"Cluster {x}"))[1])
                    
                    # Analyze search intent for each cluster
                    with st.spinner("Analyzing search intent..."):
                        from utils.intent import classify_search_intent, analyze_journey_phase
                        cluster_intents = {}
                        for cluster_id in representative_keywords.keys():
                            keywords = representative_keywords[cluster_id]
                            intent_result = classify_search_intent(keywords)
                            journey_phase = analyze_journey_phase(intent_result)
                            cluster_intents[cluster_id] = {
                                "intent": intent_result,
                                "journey_phase": journey_phase
                            }
                    
                    # Store results in session state
                    st.session_state['clustering_results'] = df
                    st.session_state['cluster_names'] = cluster_names
                    st.session_state['cluster_intents'] = cluster_intents
                    st.session_state['representative_keywords'] = representative_keywords
                    st.session_state['process_complete'] = True
                    
                    # Force refresh to show results
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
else:
        # Display results
        df = st.session_state['clustering_results']
        cluster_names = st.session_state['cluster_names']
        cluster_intents = st.session_state['cluster_intents']
        representative_keywords = st.session_state['representative_keywords']
        
        # Results tabs
        tab1, tab2, tab3 = st.tabs(["Cluster Overview", "Explore Clusters", "Export Results"])
        
        # Tab 1: Overview
        with tab1:
            st.markdown("### Clustering Results Overview")
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Keywords", len(df))
            with col2:
                st.metric("Number of Clusters", len(df['cluster_id'].unique()))
            with col3:
                avg_cluster_size = len(df) / len(df['cluster_id'].unique())
                st.metric("Average Cluster Size", f"{avg_cluster_size:.1f}")
            
            # Visualizations
            st.markdown("### Cluster Size Distribution")
            viz_col1, viz_col2 = st.columns([2, 1])
            
            with viz_col1:
                # Create cluster size visualization
                from utils.visualization import create_cluster_visualization
                cluster_size_fig = create_cluster_visualization(df)
                st.plotly_chart(cluster_size_fig, use_container_width=True)
            
            with viz_col2:
                # Search intent distribution
                from utils.visualization import create_intent_visualization
                intent_fig = create_intent_visualization(cluster_intents)
                st.plotly_chart(intent_fig, use_container_width=True)
            
            # Customer journey visualization
            st.markdown("### Customer Journey Distribution")
            
            # Count clusters in each journey phase
            journey_phases = {cluster_id: data["journey_phase"] for cluster_id, data in cluster_intents.items()}
            
            # Create a dataframe for visualization
            journey_counts = pd.Series(journey_phases).value_counts().reset_index()
            journey_counts.columns = ['phase', 'count']
            
            if not journey_counts.empty:
                journey_order = [
                    "Early (Research/Awareness)",
                    "Middle (Consideration)",
                    "Late (Decision/Purchase)",
                    "Mixed"
                ]
                
                # Create bar chart
                journey_fig = px.bar(
                    journey_counts, 
                    x='phase', 
                    y='count',
                    color='phase',
                    title="Keyword Clusters by Customer Journey Stage",
                    labels={'phase': 'Journey Stage', 'count': 'Number of Clusters'},
                    color_discrete_map={
                        "Early (Research/Awareness)": "#4CAF50",
                        "Middle (Consideration)": "#2196F3",
                        "Late (Decision/Purchase)": "#FF9800",
                        "Mixed": "#9E9E9E"
                    }
                )
                
                st.plotly_chart(journey_fig, use_container_width=True)
                
                st.markdown("""
                **Understanding the Customer Journey:**
                - **Early Stage (Research/Awareness)**: Users are learning about solutions to their problems (informational queries)
                - **Middle Stage (Consideration)**: Users are comparing options and evaluating alternatives (commercial queries)
                - **Late Stage (Decision/Purchase)**: Users are ready to make a purchase (transactional queries)
                
                Target your content and campaigns according to where users are in their journey.
                """)
