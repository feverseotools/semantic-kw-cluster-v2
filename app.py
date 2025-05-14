import streamlit as st
from semantic_clustering.app import SemanticKeywordClusterer
import pandas as pd
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt
from io import BytesIO
import base64

st.set_page_config(page_title="Semantic Keyword Clustering", layout="wide")

st.title("Semantic Keyword Clustering")
st.markdown("Group keywords based on semantic similarity, search intent, and customer journey mapping.")

# Function to create visualizations for the results
def create_cluster_visualization(embeddings_2d, labels, cluster_labels=None, width=800, height=600, dpi=100):
    """Create a visualization of the clusters"""
    try:
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        ax = fig.add_subplot(111)
        
        # Get unique labels and colors
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each cluster
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                color=colors[i],
                label=f"Cluster {label}",
                alpha=0.7,
                s=50,
                edgecolors='w',
                linewidths=0.5
            )
            
            # Add cluster label at centroid
            if len(embeddings_2d[mask]) > 0:
                centroid = embeddings_2d[mask].mean(axis=0)
                label_text = f"Cluster {label}"
                if cluster_labels and str(label) in cluster_labels:
                    label_text += f": {cluster_labels[str(label)]}"
                
                ax.text(
                    centroid[0],
                    centroid[1],
                    label_text,
                    fontsize=9,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
                )
        
        # Set plot properties
        ax.set_title("Keyword Clusters Visualization", fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Remove axes ticks for cleaner look
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Save to buffer
        buf = BytesIO()
        fig.tight_layout()
        plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return buf
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def create_cluster_size_chart(clusters, width=800, height=400, dpi=100):
    """Create a bar chart of cluster sizes"""
    try:
        # Get cluster sizes
        cluster_ids = []
        sizes = []
        
        for cluster_id, keywords in sorted(
            clusters.items(),
            key=lambda x: len(x[1]),
            reverse=True
        ):
            cluster_ids.append(cluster_id)
            sizes.append(len(keywords))
        
        # Limit to top 20 clusters if there are many
        if len(cluster_ids) > 20:
            cluster_ids = cluster_ids[:19] + ['Others']
            sizes = sizes[:19] + [sum(sizes[19:])]
        
        # Create figure
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        ax = fig.add_subplot(111)
        
        # Create bar chart
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(cluster_ids)))
        bars = ax.bar(range(len(cluster_ids)), sizes, color=colors)
        
        # Add value labels on top of bars
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.1,
                str(size),
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        # Set plot properties
        ax.set_title("Cluster Sizes", fontsize=16)
        ax.set_ylabel("Number of Keywords", fontsize=12)
        ax.set_xticks(range(len(cluster_ids)))
        ax.set_xticklabels(cluster_ids, rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Save to buffer
        buf = BytesIO()
        fig.tight_layout()
        plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return buf
    except Exception as e:
        st.error(f"Error creating cluster size chart: {str(e)}")
        return None

with st.sidebar:
    st.header("Configuration")
    
    uploaded_file = st.file_uploader("Upload Keywords", type=["csv", "txt", "json"])
    
    model_options = ["all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2", "all-mpnet-base-v2"]
    embedding_model = st.selectbox("Embedding Model", model_options)
    
    clustering_method = st.selectbox(
        "Clustering Method", 
        ["kmeans", "dbscan", "hdbscan", "agglomerative"],
        help="KMeans is recommended for most cases"
    )
    
    if clustering_method in ["kmeans", "agglomerative"]:
        n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=50, value=10)
        optimize_clusters = st.checkbox("Optimize number of clusters", value=False)
        
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
        else:  # HDBSCAN
            min_cluster_size = st.slider("Min Cluster Size", min_value=2, max_value=20, value=5)
            min_samples = st.slider("Min Samples", min_value=1, max_value=20, value=None)
        
        optimize_clusters = False
        n_clusters = None
    
    # Advanced Options - estas deben estar fuera del bloque condicional else
    st.subheader("Advanced Options")
    perform_preprocessing = st.checkbox("Preprocess keywords", value=True)
    use_batching = st.checkbox("Use batch processing for large datasets", value=False)
    batch_size = None
    
    # Control batch_size solo si use_batching está activado
    if use_batching:
        batch_size = st.number_input("Batch Size", min_value=100, max_value=10000, value=1000, step=100)
        
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
    
    export_formats = st.multiselect(
        "Export Formats", 
        ["json", "excel", "html", "pdf"],
        default=["json"]
    )

    # Create the directory for exports
    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)

if uploaded_file is not None:
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
        
        st.write(f"Total keywords loaded: {len(keywords)}")
        
        if len(keywords) > 1000:
            st.warning(f"Large number of keywords detected ({len(keywords)}). Processing may take some time.")
        
        if st.button("Start Clustering"):
            with st.spinner("Clustering keywords..."):
                progress_bar = st.progress(0)
                
                # Initialize the clusterer with appropriate parameters
                clusterer_params = {
                    "embedding_model": embedding_model,
                    "method": clustering_method,
                    "perform_preprocessing": perform_preprocessing
                }
                
                # Add method-specific parameters
                if clustering_method in ["kmeans", "agglomerative"]:
                    clusterer_params["n_clusters"] = n_clusters if not optimize_clusters else None
                elif clustering_method == "dbscan":
                    clusterer_params["eps"] = eps
                    clusterer_params["min_samples"] = min_samples
                elif clustering_method == "hdbscan":
                    clusterer_params["min_cluster_size"] = min_cluster_size
                    clusterer_params["min_samples"] = min_samples if min_samples else None
                
                # Initialize clusterer
                clusterer = SemanticKeywordClusterer(**clusterer_params)
                
                # Load keywords
                progress_bar.progress(0.1)
                clusterer.load_keywords(keywords)
                
                # Perform clustering with progress updates
                progress_bar.progress(0.3)
                
                cluster_params = {}
                if clustering_method in ["kmeans", "agglomerative"]:
                    cluster_params["optimize"] = optimize_clusters
                    if optimize_clusters:
                        cluster_params["min_clusters"] = min_clusters
                        cluster_params["max_clusters"] = max_clusters
                
                cluster_params["label_method"] = label_method
                
                # Perform clustering
                if use_batching and batch_size:
                    clusters = clusterer.batch_process(
                        batch_size=batch_size,
                        **cluster_params
                    )
                else:
                    clusters = clusterer.cluster(**cluster_params)
                
                progress_bar.progress(0.8)
                
                # Get results
                metrics = clusterer.get_metrics()
                cluster_labels = clusterer.get_cluster_labels()
                embeddings_2d, labels = clusterer.get_visualization_data()
                
                progress_bar.progress(1.0)
                
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Clusters", "Visualizations", "Export"])
                
                with tab1:
                    st.subheader("Clustering Results Overview")
                    
                    # Display key metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Clusters", len(clusters))
                    with col2:
                        st.metric("Total Keywords", sum(len(kws) for kws in clusters.values()))
                    with col3:
                        if "silhouette_score" in metrics:
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
                    
                    # Display cluster size chart
                    st.subheader("Cluster Size Distribution")
                    size_chart = create_cluster_size_chart(clusters)
                    if size_chart:
                        st.image(size_chart, use_column_width=True)
                
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
                            # Create a DataFrame for better display
                            df = pd.DataFrame(sorted(keywords), columns=["Keyword"])
                            st.dataframe(df, hide_index=True)
                            
                            # Add download option for this cluster
                            cluster_csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label=f"Download Cluster {cluster_id} as CSV",
                                data=cluster_csv,
                                file_name=f"cluster_{cluster_id}.csv",
                                mime="text/csv"
                            )
                
                with tab3:
                    st.subheader("Visualizations")
                    
                    if embeddings_2d is not None and labels is not None:
                        # Create cluster visualization
                        vis_buf = create_cluster_visualization(
                            embeddings_2d, labels, cluster_labels
                        )
                        
                        if vis_buf:
                            st.image(vis_buf, use_column_width=True)
                            
                            # Convert buffer to base64 for download
                            b64_vis = base64.b64encode(vis_buf.getvalue()).decode()
                            href = f'<a href="data:image/png;base64,{b64_vis}" download="cluster_visualization.png">Download Visualization</a>'
                            st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.info("Visualization data not available.")
                
                with tab4:
                    st.subheader("Export Results")
                    
                    # Export options
                    export_results = clusterer.save(
                        output_dir=export_dir,
                        formats=export_formats,
                        file_prefix=f"keywords_clustering_{clustering_method}"
                    )
                    
                    if export_results:
                        st.success(f"Successfully exported results in: {', '.join(export_results.keys())}")
                        
                        for format_name, file_path in export_results.items():
                            with open(file_path, "rb") as f:
                                file_data = f.read()
                                st.download_button(
                                    label=f"Download {format_name.upper()} file",
                                    data=file_data,
                                    file_name=os.path.basename(file_path),
                                    mime=f"application/{format_name}"
                                )
                    else:
                        st.error("Failed to export results. Check logs for details.")
                        
                    # Add option to download all clusters in CSV format
                    all_keywords = []
                    for cluster_id, keywords in clusters.items():
                        for keyword in keywords:
                            all_keywords.append({
                                "cluster_id": cluster_id,
                                "cluster_label": cluster_labels.get(cluster_id, ""),
                                "keyword": keyword
                            })
                    
                    all_df = pd.DataFrame(all_keywords)
                    all_csv = all_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download All Clusters as CSV",
                        data=all_csv,
                        file_name="all_clusters.csv",
                        mime="text/csv"
                    )
                    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
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
    "Semantic Keyword Clustering tool - See the [GitHub repository](https://github.com/yourusername/semantic-keyword-clustering) for more information."
)
