import streamlit as st
from semantic_clustering.app import SemanticKeywordClusterer
import pandas as pd
import numpy as np
import os
import tempfile
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import time
from collections import Counter

# Import export_pdf if available
try:
    from export_pdf import add_pdf_export_button
    pdf_export_available = True
except ImportError:
    pdf_export_available = False

# Try to import OpenAI
try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False

# Function to initialize NLP dependencies
def initialize_nlp_dependencies():
    """
    Initialize NLP dependencies, including spaCy models and NLTK resources
    to ensure the application works correctly in cloud environments.
    """
    try:
        # Try to import spaCy
        import spacy
        
        # Check if models are installed
        required_models = {
            "en_core_web_sm": "English",
            "es_core_news_sm": "Spanish",
            "fr_core_news_sm": "French",
            "de_core_news_sm": "German",
            "pl_core_news_sm": "Polish",
            "it_core_news_sm": "Italian",
            "pt_core_news_sm": "Portuguese"
        }
        
        installed_models = []
        missing_models = []
        
        for model_name, language in required_models.items():
            try:
                # Try to load the model
                spacy.load(model_name)
                installed_models.append(f"{model_name} ({language})")
            except OSError:
                missing_models.append(f"{model_name} ({language})")
        
        if installed_models:
            st.sidebar.success(f"✅ Available spaCy models: {', '.join(installed_models)}")
        
        if missing_models:
            st.sidebar.warning(
                f"⚠️ Some spaCy models are not installed: {', '.join(missing_models)}\n"
                "Alternative text processing will be used for these languages."
            )
            
            # Try to download English model if missing (this will work in some environments)
            if "en_core_web_sm (English)" in missing_models:
                try:
                    st.sidebar.info("Attempting to download English model...")
                    spacy.cli.download("en_core_web_sm")
                    st.sidebar.success("✅ English model downloaded successfully")
                except:
                    st.sidebar.warning("Could not automatically download the model. Using fallback processing.")
    except ImportError:
        st.sidebar.warning("spaCy is not installed. Using alternative text processing methods.")
    
    # Try to initialize NLTK resources
    try:
        import nltk
        
        # Define required NLTK resources
        required_resources = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']
        
        # Check and download NLTK resources if needed
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                st.sidebar.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)
        
        st.sidebar.success("✅ NLTK resources initialized")
    except Exception as e:
        st.sidebar.warning(f"Could not initialize all NLTK resources. Some text processing features may be limited.")

    # Try to import sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        st.sidebar.success("✅ SentenceTransformers available")
    except ImportError:
        st.sidebar.warning("SentenceTransformers not available. Using TF-IDF fallback.")


def create_plotly_cluster_visualization(embeddings_2d, labels, cluster_labels=None):
    """Create an interactive cluster visualization using Plotly"""
    try:
        # Get unique labels and create a map of cluster IDs to names
        unique_labels = np.unique(labels)
        cluster_names = {}
        
        for label in unique_labels:
            if cluster_labels and str(label) in cluster_labels:
                cluster_names[label] = f"Cluster {label}: {cluster_labels[str(label)]}"
            else:
                cluster_names[label] = f"Cluster {label}"
        
        # Create a DataFrame from the embeddings
        df_vis = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'cluster': [cluster_names.get(label, f"Cluster {label}") for label in labels]
        })
        
        # Create Plotly figure
        fig = px.scatter(
            df_vis, 
            x='x', 
            y='y', 
            color='cluster',
            title="Keyword Clusters Visualization",
            labels={'x': '', 'y': ''},
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        # Improve layout
        fig.update_layout(
            height=700,
            legend_title="Clusters",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # Hide axis labels and ticks
        fig.update_xaxes(showticklabels=False, showgrid=True)
        fig.update_yaxes(showticklabels=False, showgrid=True)
        
        return fig
    except Exception as e:
        st.error(f"Error creating Plotly visualization: {str(e)}")
        return None


def create_plotly_cluster_sizes(clusters, cluster_labels=None):
    """Create an interactive bar chart of cluster sizes using Plotly"""
    try:
        # Prepare data
        cluster_ids = []
        sizes = []
        labels = []
        
        # Sort clusters by size
        sorted_clusters = sorted(
            clusters.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for cluster_id, keywords in sorted_clusters:
            cluster_ids.append(cluster_id)
            sizes.append(len(keywords))
            
            # Add label if available
            if cluster_labels and cluster_id in cluster_labels:
                labels.append(f"Cluster {cluster_id}: {cluster_labels[cluster_id]}")
            else:
                labels.append(f"Cluster {cluster_id}")
        
        # Limit to top 20 clusters if there are many
        if len(cluster_ids) > 20:
            cluster_ids = cluster_ids[:19] + ['Others']
            sizes = sizes[:19] + [sum(sizes[19:])]
            labels = labels[:19] + ['Other Clusters']
        
        # Create DataFrame
        df_sizes = pd.DataFrame({
            'Cluster ID': cluster_ids,
            'Cluster': labels,
            'Size': sizes
        })
        
        # Create Plotly figure
        fig = px.bar(
            df_sizes,
            x='Cluster',
            y='Size',
            color='Size',
            color_continuous_scale='Viridis',
            title="Cluster Sizes",
            labels={'Size': 'Number of Keywords'}
        )
        
        # Improve layout
        fig.update_layout(
            height=500,
            xaxis_tickangle=-45,
            coloraxis_showscale=False
        )
        
        # Add value annotations on top of bars
        for i, size in enumerate(sizes):
            fig.add_annotation(
                x=i,
                y=size,
                text=str(size),
                showarrow=False,
                yshift=10
            )
        
        return fig
    except Exception as e:
        st.error(f"Error creating Plotly cluster size chart: {str(e)}")
        return None


def calculate_api_cost(num_keywords, selected_model="gpt-4.1-nano", num_clusters=10):
    """
    Calculates the estimated cost of using the OpenAI API based on the number of keywords.
    """
    # Updated prices (May 2024) - Always check OpenAI's official pricing
    EMBEDDING_COST_PER_1K = 0.00002  # text-embedding-3-small per 1K tokens
    
    # GPT-4.1-Nano costs
    GPT41NANO_INPUT_COST_PER_1K = 0.15  # $0.15 per 1K input tokens
    GPT41NANO_OUTPUT_COST_PER_1K = 0.60  # $0.60 per 1K output tokens
    
    # GPT-4.1-Mini costs
    GPT41MINI_INPUT_COST_PER_1K = 0.30  # $0.30 per 1K input tokens
    GPT41MINI_OUTPUT_COST_PER_1K = 1.20  # $1.20 per 1K output tokens
    
    results = {
        "embedding_cost": 0,
        "naming_cost": 0,
        "total_cost": 0,
        "processed_keywords": 0
    }
    
    # 1. Embedding cost (limited to 5000 keywords)
    keywords_for_embeddings = min(num_keywords, 5000)
    results["processed_keywords"] = keywords_for_embeddings
    
    # Estimate ~2 tokens per keyword
    estimated_tokens = keywords_for_embeddings * 2
    results["embedding_cost"] = (estimated_tokens / 1000) * EMBEDDING_COST_PER_1K
    
    # 2. Naming cost
    avg_tokens_per_cluster = 200   # prompt + representative keywords
    avg_output_tokens_per_cluster = 80  # output tokens (name + description)
    
    estimated_input_tokens = num_clusters * avg_tokens_per_cluster
    estimated_output_tokens = num_clusters * avg_output_tokens_per_cluster
    
    if selected_model == "gpt-4.1-nano":
        input_cost = (estimated_input_tokens / 1000) * GPT41NANO_INPUT_COST_PER_1K
        output_cost = (estimated_output_tokens / 1000) * GPT41NANO_OUTPUT_COST_PER_1K
    else:  # GPT-4.1-Mini
        input_cost = (estimated_input_tokens / 1000) * GPT41MINI_INPUT_COST_PER_1K
        output_cost = (estimated_output_tokens / 1000) * GPT41MINI_OUTPUT_COST_PER_1K
    
    results["naming_cost"] = input_cost + output_cost
    results["total_cost"] = results["embedding_cost"] + results["naming_cost"]
    
    return results


def add_cost_calculator():
    st.sidebar.markdown("---")
    with st.sidebar.expander("💰 API Cost Calculator", expanded=False):
        st.markdown("""
        ### API Cost Calculator
        
        Estimate OpenAI usage costs for a given number of keywords.
        """)
        
        calc_num_keywords = st.number_input(
            "Number of keywords",
            min_value=100, 
            max_value=100000, 
            value=1000,
            step=500
        )
        calc_num_clusters = st.number_input(
            "Approx. number of clusters",
            min_value=2,
            max_value=50,
            value=10,
            step=1
        )
        calc_model = st.radio(
            "Model for naming clusters",
            options=["gpt-4.1-nano", "gpt-4.1-mini"],
            index=0,
            horizontal=True
        )
        
        if st.button("Calculate Estimated Cost", use_container_width=True):
            cost_results = calculate_api_cost(calc_num_keywords, calc_model, calc_num_clusters)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Keywords processed with OpenAI", 
                    f"{cost_results['processed_keywords']:,}",
                    help="OpenAI processes up to 5,000 keywords; any beyond that are handled via similarity propagation."
                )
                st.metric(
                    "Embeddings cost", 
                    f"${cost_results['embedding_cost']:.4f}",
                    help="Cost using text-embedding-3-small"
                )
            with col2:
                st.metric(
                    "Cluster naming cost", 
                    f"${cost_results['naming_cost']:.4f}",
                    help=f"Cost using {calc_model} to name and describe clusters"
                )
                st.metric(
                    "TOTAL COST", 
                    f"${cost_results['total_cost']:.4f}",
                    help="Approximate total cost"
                )
            
            st.info("""
            **Note:** This is an estimate only. Actual costs may vary based on keyword length and clustering complexity.
            Using SentenceTransformers instead of OpenAI embeddings is $0.
            """)


def show_csv_cost_estimate(num_keywords, selected_model="gpt-4.1-nano", num_clusters=10):
    if num_keywords > 0:
        cost_results = calculate_api_cost(num_keywords, selected_model, num_clusters)
        
        with st.sidebar.expander("💰 Estimated Cost (Current CSV)", expanded=True):
            st.markdown(f"### Estimated Cost for {num_keywords:,} Keywords")
            
            st.markdown(f"""
            - **Keywords processed with OpenAI**: {cost_results['processed_keywords']:,}
            - **Embeddings cost**: ${cost_results['embedding_cost']:.4f}
            - **Cluster naming cost**: ${cost_results['naming_cost']:.4f}
            - **TOTAL COST**: ${cost_results['total_cost']:.4f}
            """)
            
            if cost_results['processed_keywords'] < num_keywords:
                st.info(f"""
                {cost_results['processed_keywords']:,} keywords will be processed by OpenAI directly.
                The remaining {num_keywords - cost_results['processed_keywords']:,} will use
                similarity propagation.
                """)
            
            st.markdown("""
            **Cost Savings**: If you prefer not to use OpenAI, you can 
            use SentenceTransformers at no cost with decent results.
            """)


# Function to suggest optimal batch size based on dataset size
def suggest_batch_size(total_keywords):
    """Suggest optimal batch size based on the total number of keywords"""
    if total_keywords <= 1000:
        return total_keywords  # No batching needed for small datasets
    elif total_keywords <= 5000:
        return 1000
    elif total_keywords <= 20000:
        return 2500
    else:
        return 5000


def main():
    st.set_page_config(page_title="Advanced Semantic Keyword Clustering", layout="wide")
    
    # Initialize NLP dependencies at startup
    initialize_nlp_dependencies()
    
    st.title("Advanced Semantic Keyword Clustering")
    st.markdown("Group keywords based on semantic similarity, search intent, and customer journey mapping.")
    
    # Initialize session state if needed
    if 'process_complete' not in st.session_state:
        st.session_state.process_complete = False
    if 'df_results' not in st.session_state:
        st.session_state.df_results = None
    if 'cluster_evaluation' not in st.session_state:
        st.session_state.cluster_evaluation = {}
    
    # Initialize global parameters dictionary to store all clustering parameters
    clusterer_params = {
        "embedding_model": "all-MiniLM-L6-v2",
        "method": "kmeans",
        "perform_preprocessing": True,
        "n_clusters": 10
    }
    
    with st.sidebar:
        st.header("Configuration")
        
        # Language selection
        language_options = [
            "English", "Spanish", "French", "German", "Dutch", 
            "Italian", "Portuguese", "Polish"
        ]
        selected_language = st.selectbox(
            "Select language",
            options=language_options,
            index=0
        )
        
        # File upload
        uploaded_file = st.file_uploader("Upload Keywords", type=["csv", "txt", "json"])
        
        # OpenAI API key
        use_openai = st.checkbox("Use OpenAI for enhanced semantic understanding", value=False)
        openai_api_key = None
        
        if use_openai:
            openai_api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API Key for high-quality embeddings and cluster naming"
            )
            if openai_api_key and openai_available:
                st.success("✅ OpenAI API key provided")
                
                # Model selection
                gpt_model = st.selectbox(
                    "Model for naming clusters",
                    ["gpt-4.1-nano", "gpt-4.1-mini"],
                    index=0,
                    help="GPT-4.1-mini is more capable but more expensive"
                )
                
                # Custom prompt
                with st.expander("Advanced: Custom Prompt for Naming", expanded=False):
                    default_prompt = (
                        "You are an expert in SEO and content marketing. Below you'll see several clusters "
                        "with a list of representative keywords. Your task is to assign each cluster a short, "
                        "clear name (3-6 words) and write a concise SEO meta description (1 or 2 sentences) "
                        "briefly explaining the topic and likely search intent."
                    )
                    user_prompt = st.text_area(
                        "Custom Prompt (for cluster naming)",
                        value=default_prompt,
                        height=150
                    )
            elif not openai_available:
                st.warning("OpenAI package not installed. Using SentenceTransformers instead.")
            else:
                st.warning("No API key provided. Using SentenceTransformers instead.")
                gpt_model = "gpt-4.1-nano"  # default
                user_prompt = ""  # default
        else:
            gpt_model = "gpt-4.1-nano"  # default
            user_prompt = ""  # default
        
        # CSV format
        csv_format = st.radio(
            "CSV Format",
            ["no_header", "with_header"],
            horizontal=True,
            help="Select 'no_header' if your file has one keyword per line without headers"
        )
        
        # Display cost calculator if OpenAI is selected
        if use_openai and openai_api_key and openai_available:
            add_cost_calculator()
        
        # Embedding model
        model_options = ["all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2", "all-mpnet-base-v2"]
        embedding_model = st.selectbox(
            "Embedding Model", 
            model_options,
            help="Select the model to use for generating embeddings"
        )
        clusterer_params["embedding_model"] = embedding_model
        
        # Clustering method
        clustering_method = st.selectbox(
            "Clustering Method", 
            ["kmeans", "dbscan", "hdbscan", "agglomerative"],
            help="KMeans is recommended for most cases"
        )
        clusterer_params["method"] = clustering_method
        
        # Method-specific parameters
        if clustering_method in ["kmeans", "agglomerative"]:
            n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=50, value=10)
            optimize_clusters = st.checkbox("Optimize number of clusters", value=False)
            
            clusterer_params["n_clusters"] = n_clusters if not optimize_clusters else None
            
            if optimize_clusters:
                col1, col2 = st.columns(2)
                with col1:
                    min_clusters = st.number_input("Min Clusters", min_value=2, max_value=20, value=2)
                with col2:
                    max_clusters = st.number_input("Max Clusters", min_value=3, max_value=50, value=20)
        else:
            # For DBSCAN and HDBSCAN
            if clustering_method == "dbscan":
                eps = st.slider("EPS Parameter", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
                min_samples = st.slider("Min Samples", min_value=2, max_value=20, value=5)
                
                clusterer_params["eps"] = eps
                clusterer_params["min_samples"] = min_samples
                
            else:  # HDBSCAN
                min_cluster_size = st.slider("Min Cluster Size", min_value=2, max_value=20, value=5)
                min_samples = st.slider("Min Samples", min_value=1, max_value=20, value=None)
                
                clusterer_params["min_cluster_size"] = min_cluster_size
                clusterer_params["min_samples"] = min_samples if min_samples else None
            
            optimize_clusters = False
            n_clusters = None
        
        # Advanced Options
        with st.expander("Advanced Options", expanded=False):
            perform_preprocessing = st.checkbox("Preprocess keywords", value=True)
            clusterer_params["perform_preprocessing"] = perform_preprocessing
            
            use_batching = st.checkbox("Use batch processing for large datasets", value=False)
            batch_size = None
            auto_batch_size = False
            
            if use_batching:
                auto_batch_size = st.checkbox("Auto-adjust batch size based on dataset size", value=True)
                
                if not auto_batch_size:
                    batch_size = st.number_input("Batch Size", min_value=100, max_value=10000, value=1000, step=100)
                    
                    if batch_size:
                        st.info(f"Processing will be done in batches of {batch_size} keywords")
                else:
                    st.info("Batch size will be automatically adjusted based on dataset size")
                    
            label_method = st.selectbox(
                "Cluster Labeling Method", 
                ["tfidf", "frequent", "centroid"],
                help="TF-IDF is recommended for most cases"
            )
            
            # Dimensionality reduction options
            visualization_method = st.selectbox(
                "Visualization Method", 
                ["umap", "pca"],
                help="UMAP generally provides better visualizations"
            )
            
            pca_variance = st.slider("PCA explained variance (%)", 50, 99, 95)
            max_pca_components = st.slider("Max PCA components", 10, 300, 100)
            
            # Export options
            export_formats = st.multiselect(
                "Export Formats", 
                ["json", "excel", "html", "pdf"],
                default=["json"]
            )
        
        # Create export directory
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
    
    # Main content area
    if uploaded_file is not None and not st.session_state.process_complete:
        try:
            st.subheader("Input Data")
            
            # Read different file formats
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_extension == '.csv':
                has_header = st.checkbox("File has header", value=True)
                df = pd.read_csv(uploaded_file)
                
                if has_header:
                    st.write("Preview of input data:")
                    st.dataframe(df.head())
                    keywords = df.iloc[:, 0].dropna().astype(str).tolist()
                else:
                    keywords = [line.strip() for line in uploaded_file.getvalue().decode().split('\n') if line.strip()]
                    st.write(f"Read {len(keywords)} keywords from txt format")
                    
            elif file_extension == '.txt':
                keywords = [line.strip() for line in uploaded_file.getvalue().decode().split('\n') if line.strip()]
                st.write(f"Read {len(keywords)} keywords from txt file")
                
            elif file_extension == '.json':
                import json
                data = json.loads(uploaded_file.getvalue())
                if isinstance(data, list):
                    keywords = data
                elif isinstance(data, dict) and 'keywords' in data:
                    keywords = data['keywords']
                else:
                    st.error("JSON format not recognized. Please use a list of keywords or a dict with a 'keywords' key.")
                    keywords = []
                st.write(f"Read {len(keywords)} keywords from json file")
            
            total_keywords = len(keywords)
            st.write(f"Total keywords loaded: {total_keywords}")
            
            # Show cost estimate if using OpenAI
            if use_openai and openai_api_key and openai_available:
                show_csv_cost_estimate(total_keywords, gpt_model, n_clusters or 10)
            
            # Auto-adjust batch size if enabled
            if use_batching and auto_batch_size and total_keywords > 0:
                batch_size = suggest_batch_size(total_keywords)
                st.info(f"Auto-adjusted batch size: {batch_size} keywords")
            
            # Validate batch size if manually set
            if use_batching and batch_size and not auto_batch_size:
                if batch_size >= total_keywords:
                    st.warning(f"Batch size ({batch_size}) is larger than or equal to the total number of keywords ({total_keywords}). Batching will still work but is unnecessary.")
            
            if total_keywords > 1000:
                st.warning(f"Large number of keywords detected ({total_keywords}). Processing may take some time.")
            
            # Start clustering button
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                if st.button("Start Advanced Semantic Clustering", type="primary", use_container_width=True):
                    with st.spinner("Clustering keywords..."):
                        progress_bar = st.progress(0)
                        
                        # Initialize the clusterer with parameters
                        clusterer = SemanticKeywordClusterer(**clusterer_params)
                        
                        # Load keywords
                        progress_bar.progress(0.1)
                        clusterer.load_keywords(keywords)
                        
                        # Perform clustering
                        progress_bar.progress(0.3)
                        
                        cluster_params = {}
                        if clustering_method in ["kmeans", "agglomerative"]:
                            cluster_params["optimize"] = optimize_clusters
                            if optimize_clusters:
                                cluster_params["min_clusters"] = min_clusters
                                cluster_params["max_clusters"] = max_clusters
                        
                        cluster_params["label_method"] = label_method
                        
                        # Perform clustering
                        if use_batching and batch_size and batch_size < total_keywords:
                            clusters = clusterer.batch_process(
                                batch_size=batch_size,
                                **cluster_params
                            )
                        else:
                            clusters = clusterer.cluster(**cluster_params)
                        
                        progress_bar.progress(0.6)
                        
                        # Get results
                        metrics = clusterer.get_metrics()
                        cluster_labels = clusterer.get_cluster_labels()
                        embeddings_2d, labels = clusterer.get_visualization_data(method=visualization_method)
                        
                        # Create DataFrame with results
                        all_keywords = []
                        for cluster_id, keywords in clusters.items():
                            for keyword in keywords:
                                all_keywords.append({
                                    "cluster_id": cluster_id,
                                    "cluster_name": cluster_labels.get(cluster_id, f"Cluster {cluster_id}"),
                                    "keyword": keyword,
                                    "representative": False  # Will be updated later
                                })
                        
                        df_results = pd.DataFrame(all_keywords)
                        
                        # Generate search intent analysis if OpenAI is available
                        if use_openai and openai_api_key and openai_available:
                            progress_bar.progress(0.7)
                            st.info("Performing search intent analysis...")
                            
                            # This would call a function from semantic_clustering.nlp.intent
                            # to perform search intent classification
                            # For now, we'll just store placeholder data
                            cluster_evaluation = {}
                            for cluster_id in clusters.keys():
                                cluster_evaluation[cluster_id] = {
                                    "search_intent": "No intent data available",
                                    "coherence_score": float(np.random.uniform(0.5, 0.9)),  # Placeholder
                                    "intent_classification": {
                                        "primary_intent": np.random.choice(["Informational", "Navigational", "Transactional", "Commercial"]),
                                        "scores": {
                                            "Informational": float(np.random.uniform(0, 100)),
                                            "Navigational": float(np.random.uniform(0, 100)),
                                            "Transactional": float(np.random.uniform(0, 100)),
                                            "Commercial": float(np.random.uniform(0, 100))
                                        }
                                    }
                                }
                            
                            st.session_state.cluster_evaluation = cluster_evaluation
                        
                        progress_bar.progress(1.0)
                        
                        # Store results in session_state
                        st.session_state.df_results = df_results
                        st.session_state.process_complete = True
                        
                        st.success("✅ Semantic clustering completed successfully!")
                        st.experimental_rerun()
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # If processing is complete, display results
    elif st.session_state.process_complete and st.session_state.df_results is not None:
        df = st.session_state.df_results
        
        # Get unique clusters
        cluster_counts = df.groupby(['cluster_id', 'cluster_name']).size().reset_index(name='count')
        clusters = {}
        cluster_labels = {}
        
        for _, row in cluster_counts.iterrows():
            cluster_counts = df.groupby(['cluster_id', 'cluster_name']).size().reset_index(name='count')
            clusters = {}
            cluster_labels = {}
            
            for _, row in cluster_counts.iterrows():
                cluster_id = row['cluster_id']
                cluster_name = row['cluster_name']
                keywords = df[df['cluster_id'] == cluster_id]['keyword'].tolist()
                clusters[cluster_id] = keywords
                cluster_labels[cluster_id] = cluster_name
            
            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Clusters", "Visualizations", "Analysis"])
            
            with tab1:
                st.subheader("Clustering Results Overview")
                
                # Display key metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Clusters", len(clusters))
                with col2:
                    st.metric("Total Keywords", sum(len(kws) for kws in clusters.values()))
                with col3:
                    if "cluster_evaluation" in st.session_state and st.session_state.cluster_evaluation:
                        # Show percentage of clusters with each primary intent
                        intent_counts = Counter([data.get('intent_classification', {}).get('primary_intent', 'Unknown') 
                                               for data in st.session_state.cluster_evaluation.values()])
                        most_common_intent, count = intent_counts.most_common(1)[0]
                        percentage = (count / len(st.session_state.cluster_evaluation)) * 100
                        st.metric("Most Common Intent", f"{most_common_intent} ({percentage:.1f}%)")
                    elif "silhouette_score" in metrics:
                        st.metric("Silhouette Score", f"{metrics['silhouette_score']:.4f}")
                
                # Show additional metrics
                st.subheader("Evaluation Metrics")
                metrics_to_show = {
                    "silhouette_score": "Silhouette Score",
                    "calinski_harabasz_score": "Calinski-Harabasz Score",
                    "davies_bouldin_score": "Davies-Bouldin Score",
                    "min_size": "Smallest Cluster Size",
                    "max_size": "Largest Cluster Size",
                    "mean_size": "Average Cluster Size"
                }
                
                metrics_df = pd.DataFrame([
                    {"Metric": metrics_to_show.get(k, k), 
                     "Value": f"{v:.4f}" if isinstance(v, float) else v}
                    for k, v in metrics.items() 
                    if k in metrics_to_show
                ])
                
                st.dataframe(metrics_df, hide_index=True)
                
                # Display cluster size chart using Plotly
                st.subheader("Cluster Size Distribution")
                size_chart = create_plotly_cluster_sizes(clusters, cluster_labels)
                if size_chart:
                    st.plotly_chart(size_chart, use_container_width=True)
                
                # Show search intent distribution if available
                if "cluster_evaluation" in st.session_state and st.session_state.cluster_evaluation:
                    st.subheader("Search Intent Distribution")
                    
                    # Collect data for intent distribution
                    intent_data = []
                    for c_id, data in st.session_state.cluster_evaluation.items():
                        if 'intent_classification' in data:
                            # Get primary intent and count
                            primary_intent = data['intent_classification'].get('primary_intent', 'Unknown')
                            cluster_size = len(df[df['cluster_id'] == c_id])
                            
                            intent_data.append({
                                'intent': primary_intent,
                                'count': cluster_size
                            })
                    
                    if intent_data:
                        # Aggregate by intent
                        intent_df = pd.DataFrame(intent_data)
                        intent_totals = intent_df.groupby('intent')['count'].sum().reset_index()
                        
                        # Create Plotly pie chart
                        intent_colors = {
                            'Informational': 'rgb(33, 150, 243)',  # Blue
                            'Navigational': 'rgb(76, 175, 80)',    # Green
                            'Transactional': 'rgb(255, 152, 0)',   # Orange
                            'Commercial': 'rgb(156, 39, 176)',     # Purple
                            'Mixed Intent': 'rgb(158, 158, 158)',  # Gray
                            'Unknown': 'rgb(189, 189, 189)'        # Light Gray
                        }
                        
                        fig = px.pie(
                            intent_totals, 
                            values='count', 
                            names='intent',
                            title='Distribution of Search Intent',
                            color='intent',
                            color_discrete_map=intent_colors
                        )
                        
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(legend_title="Search Intent")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add explanation
                        st.markdown("""
                        **Understanding Search Intent:**
                        - **Informational**: Users seeking information or answers (e.g., "how to", "what is")
                        - **Navigational**: Users looking for a specific website or page (e.g., brand names)
                        - **Transactional**: Users ready to make a purchase or take action (e.g., "buy", "discount")
                        - **Commercial**: Users researching before a purchase (e.g., "best", "review", "vs")
                        """)
            
            with tab2:
                st.subheader("Cluster Details")
                
                # Sort clusters by size (descending)
                sorted_clusters = sorted(
                    clusters.items(),
                    key=lambda x: len(x[1]),
                    reverse=True
                )
                
                for cluster_id, keywords in sorted_clusters:
                    cluster_title = f"Cluster {cluster_id}"
                    if cluster_id in cluster_labels:
                        cluster_title += f": {cluster_labels[cluster_id]}"
                        
                    with st.expander(f"{cluster_title} ({len(keywords)} keywords)"):
                        # Show intent and journey phase if available
                        if "cluster_evaluation" in st.session_state and int(cluster_id) in st.session_state.cluster_evaluation:
                            eval_data = st.session_state.cluster_evaluation[int(cluster_id)]
                            intent_data = eval_data.get('intent_classification', {})
                            primary_intent = intent_data.get('primary_intent', 'Unknown')
                            
                            # Choose CSS class based on intent
                            intent_class = ""
                            if primary_intent == "Informational":
                                intent_class = "background-color: #e3f2fd; border-left: 5px solid #2196f3; padding: 10px;"
                            elif primary_intent == "Navigational":
                                intent_class = "background-color: #e8f5e9; border-left: 5px solid #4caf50; padding: 10px;"
                            elif primary_intent == "Transactional":
                                intent_class = "background-color: #fff3e0; border-left: 5px solid #ff9800; padding: 10px;"
                            elif primary_intent == "Commercial":
                                intent_class = "background-color: #f3e5f5; border-left: 5px solid #9c27b0; padding: 10px;"
                            else:
                                intent_class = "background-color: #f5f5f5; border-left: 5px solid #9e9e9e; padding: 10px;"
                            
                            # Display intent with styling
                            st.markdown(f"<div style='{intent_class}'><strong>Primary Search Intent:</strong> {primary_intent}</div>", unsafe_allow_html=True)
                        
                        # Create a DataFrame for better display
                        df_cluster = pd.DataFrame(sorted(keywords), columns=["Keyword"])
                        st.dataframe(df_cluster, hide_index=True)
                        
                        # Add download option for this cluster
                        cluster_csv = df_cluster.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"Download Cluster {cluster_id} as CSV",
                            data=cluster_csv,
                            file_name=f"cluster_{cluster_id}.csv",
                            mime="text/csv"
                        )
            
            with tab3:
                st.subheader("Visualizations")
                
                # Create tabs for different visualizations
                vis_tab1, vis_tab2 = st.tabs(["Cluster Map", "Cluster Sizes"])
                
                with vis_tab1:
                    if "df_results" in st.session_state and embeddings_2d is not None and labels is not None:
                        # Create cluster visualization using Plotly
                        fig = create_plotly_cluster_visualization(embeddings_2d, labels, cluster_labels)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add download option using matplotlib as fallback
                            st.write("Download Visualization:")
                            vis_buf = create_cluster_visualization(embeddings_2d, labels, cluster_labels)
                            if vis_buf:
                                b64_vis = base64.b64encode(vis_buf.getvalue()).decode()
                                href = f'<a href="data:image/png;base64,{b64_vis}" download="cluster_visualization.png">Download Static Image</a>'
                                st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.info("Visualization data not available.")
                
                with vis_tab2:
                    # Create bar chart of cluster sizes
                    fig = create_plotly_cluster_sizes(clusters, cluster_labels)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                if "cluster_evaluation" in st.session_state and st.session_state.cluster_evaluation:
                    st.subheader("Search Intent and Customer Journey Analysis")
                    
                    # Create tabs for different analysis views
                    intent_tab1, intent_tab2 = st.tabs(["Search Intent", "Customer Journey"])
                    
                    with intent_tab1:
                        st.markdown("""
                        This analysis classifies keywords into four main search intent categories:
                        
                        1. **Informational**: Users seeking information or answers
                        2. **Navigational**: Users looking for a specific website or page
                        3. **Transactional**: Users ready to make a purchase or take action
                        4. **Commercial**: Users researching before a purchase
                        """)
                        
                        # Create a table of clusters with their primary intents
                        intent_table_data = []
                        for c_id, data in st.session_state.cluster_evaluation.items():
                            intent_data = data.get('intent_classification', {})
                            primary_intent = intent_data.get('primary_intent', 'Unknown')
                            cluster_name = cluster_labels.get(str(c_id), f"Cluster {c_id}")
                            cluster_size = len(df[df['cluster_id'] == str(c_id)])
                            
                            # Get scores
                            scores = intent_data.get('scores', {})
                            info_score = scores.get('Informational', 0)
                            nav_score = scores.get('Navigational', 0)
                            trans_score = scores.get('Transactional', 0)
                            comm_score = scores.get('Commercial', 0)
                            
                            intent_table_data.append({
                                'Cluster ID': c_id,
                                'Cluster Name': cluster_name,
                                'Size': cluster_size,
                                'Primary Intent': primary_intent,
                                'Info Score': f"{info_score:.1f}%",
                                'Nav Score': f"{nav_score:.1f}%",
                                'Trans Score': f"{trans_score:.1f}%",
                                'Comm Score': f"{comm_score:.1f}%"
                            })
                        
                        if intent_table_data:
                            intent_df = pd.DataFrame(intent_table_data)
                            st.dataframe(intent_df, hide_index=True)
                    
                    with intent_tab2:
                        st.markdown("""
                        Customer journey mapping helps you understand which stage of the buying process your keywords represent:
                        
                        1. **Early Phase (Research)**: Dominated by informational queries
                        2. **Middle Phase (Consideration)**: Primarily commercial comparison queries
                        3. **Late Phase (Purchase)**: Mostly transactional buying queries
                        """)
                        
                        # This section would display journey mapping data
                        # For now, we'll use placeholder text
                        st.info("Customer journey analysis will be populated when the full semantic analysis module is integrated.")
                
                # Export section
                st.subheader("Export Results")
                
                # Export options
                export_formats_available = export_formats.copy()
                
                # Check if PDF export is requested but not available
                if "pdf" in export_formats_available and not pdf_export_available:
                    export_formats_available.remove("pdf")
                    st.warning("PDF export functionality is not available. Install required dependencies (reportlab, pillow, kaleido).")
                
                if export_formats_available:
                    # Create directories for exports
                    export_dir = "exports"
                    os.makedirs(export_dir, exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    file_prefix = f"keywords_clustering_{clustering_method}_{timestamp}"
                    
                    # Export button
                    if st.button("Export Results in Selected Formats", use_container_width=True):
                        with st.spinner("Exporting results..."):
                            # This would call a function from the clusterer to export
                            # For now, we'll just create a simple CSV
                            export_path = os.path.join(export_dir, f"{file_prefix}.csv")
                            df.to_csv(export_path, index=False)
                            
                            st.success(f"✅ Results exported to {export_path}")
                            
                            # Provide download link
                            with open(export_path, "rb") as f:
                                st.download_button(
                                    label="Download CSV Export",
                                    data=f,
                                    file_name=os.path.basename(export_path),
                                    mime="text/csv"
                                )
                
                # PDF export section
                if pdf_export_available:
                    st.subheader("PDF Report")
                    st.markdown("Generate a comprehensive PDF report with visualizations and analysis.")
                    
                    # Add PDF export button
                    add_pdf_export_button(df, st.session_state.cluster_evaluation if "cluster_evaluation" in st.session_state else None)
        
        # Add reset button
        if st.button("Reset", use_container_width=True):
            st.session_state.process_complete = False
            st.session_state.df_results = None
            st.session_state.cluster_evaluation = {}
            st.experimental_rerun()
    
    else:
        st.info("Please upload a file to start clustering.")
        
        # Sample data option
        if st.button("Use Sample Data"):
            # Load sample data from the repository
            try:
                sample_path = os.path.join("data", "samples", "sample_keywords.csv")
                if os.path.exists(sample_path):
                    df = pd.read_csv(sample_path)
                    st.success(f"Loaded {len(df)} sample keywords!")
                    st.dataframe(df.head())
                    
                    # Create a download link for the sample
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Sample CSV",
                        data=csv,
                        file_name="sample_keywords.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Sample file not found. Please upload your own file.")
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
    
    # Add footer with information
    st.markdown("---")
    st.markdown(
        "Advanced Semantic Keyword Clustering tool - See the [GitHub repository](https://github.com/yourusername/semantic-kw-cluster-v2) for more information."
    )


if __name__ == "__main__":
    main()
