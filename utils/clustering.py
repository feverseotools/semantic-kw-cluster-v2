import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

@st.cache_data
def cluster_keywords(embeddings, num_clusters=10, pca_variance=0.95):
    """
    Cluster keywords based on their embeddings.
    
    Args:
        embeddings (numpy.ndarray): Array of keyword embeddings
        num_clusters (int): Number of clusters to create
        pca_variance (float): Explained variance ratio for PCA dimensionality reduction
        
    Returns:
        dict: Dictionary containing cluster labels and other clustering outputs
    """
    if embeddings.shape[0] == 0:
        st.error("No embeddings to cluster")
        return {"labels": np.array([])}
    
    # Apply PCA for dimensionality reduction if needed
    if embeddings.shape[1] > 50:  # If embeddings have high dimensionality
        embeddings_reduced = apply_pca(embeddings, pca_variance)
    else:
        embeddings_reduced = embeddings
    
    # Apply KMeans clustering
    try:
        with st.spinner(f"Creating {num_clusters} clusters..."):
            # Ensure num_clusters doesn't exceed number of samples
            effective_num_clusters = min(num_clusters, embeddings.shape[0])
            
            # Run KMeans
            kmeans = KMeans(
                n_clusters=effective_num_clusters,
                random_state=42,
                n_init=10  # Multiple initializations to find best clustering
            )
            
            cluster_labels = kmeans.fit_predict(embeddings_reduced)
            
            # KMeans labels start at 0, add 1 for more human-readable labels
            cluster_labels = cluster_labels + 1
            
            # Calculate cluster centers
            cluster_centers = kmeans.cluster_centers_
            
            # Calculate inertia (within-cluster sum of squares)
            inertia = kmeans.inertia_
        
        st.success(f"✅ Created {effective_num_clusters} clusters successfully")
        
        return {
            "labels": cluster_labels,
            "centers": cluster_centers,
            "inertia": inertia,
            "pca_embedding": embeddings_reduced
        }
    
    except Exception as e:
        st.error(f"Error during clustering: {str(e)}")
        # Return random clusters as fallback
        random_labels = np.random.randint(1, num_clusters + 1, size=embeddings.shape[0])
        return {"labels": random_labels}

@st.cache_data
def apply_pca(embeddings, explained_variance_ratio=0.95):
    """
    Apply PCA for dimensionality reduction.
    
    Args:
        embeddings (numpy.ndarray): Original embeddings
        explained_variance_ratio (float): Minimum explained variance to preserve
        
    Returns:
        numpy.ndarray: Reduced embeddings
    """
    try:
        with st.spinner("Reducing dimensionality with PCA..."):
            # First, fit PCA to determine optimal number of components
            pca_analyzer = PCA()
            pca_analyzer.fit(embeddings)
            
            # Find number of components needed to explain desired variance
            cumulative_variance = np.cumsum(pca_analyzer.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= explained_variance_ratio) + 1
            
            # Apply PCA with determined number of components
            pca = PCA(n_components=n_components)
            reduced_embeddings = pca.fit_transform(embeddings)
            
            st.info(f"Reduced dimensions from {embeddings.shape[1]} to {n_components} while preserving {explained_variance_ratio*100:.1f}% of variance")
            
            return reduced_embeddings
    
    except Exception as e:
        st.error(f"Error applying PCA: {str(e)}")
        return embeddings  # Return original embeddings if PCA fails
