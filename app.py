# app.py

import streamlit as st
import pandas as pd

# --- Page Configuration ---
# Set the page title and layout
# The layout="wide" option uses the full page width
st.set_page_config(
    page_title="Semantic Keyword Clustering Tool",
    layout="wide",
    initial_sidebar_state="expanded" # Optional: Keep the sidebar open initially
)

# --- Application Title ---
st.title("Semantic Keyword Clustering Tool 🚀")
st.markdown("""
Welcome! This tool helps you cluster keywords based on semantic relevance using OpenAI embeddings.
Upload your keyword list in CSV format, provide your OpenAI API key, and configure the clustering options below.
""")

# --- Sidebar for Inputs ---
st.sidebar.header("Configuration")

# API Key Input
# Use type="password" to mask the input
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password", key="api_key_input")

# File Uploader
# Allows users to upload a CSV file
# type=["csv"] restricts uploads to CSV files only
uploaded_file = st.sidebar.file_uploader("Upload your Keyword CSV file", type=["csv"], key="file_uploader")

# --- Main Area for Processing and Results ---

# Placeholder for CSV column selection (if file is uploaded)
keyword_column = None
if uploaded_file is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("CSV Configuration")
    try:
        # Read only the first few rows to get headers without loading the whole file yet
        df_peek = pd.read_csv(uploaded_file, nrows=5)
        # Reset file pointer to the beginning for later full read
        uploaded_file.seek(0)
        
        # Let user select the keyword column
        keyword_column = st.sidebar.selectbox(
            "Select the column containing keywords:",
            options=df_peek.columns,
            index=0, # Default to the first column
            key="keyword_column_selector"
        )
        st.sidebar.info(f"Selected keyword column: **{keyword_column}**")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV headers: {e}")
        uploaded_file = None # Reset uploaded file if error

# Placeholder for Model Selection
st.sidebar.markdown("---")
st.sidebar.subheader("Clustering Options")
embedding_model = st.sidebar.selectbox(
    "Select OpenAI Embedding Model:",
    options=["text-embedding-3-small", "text-embedding-3-large"],
    index=0, # Default to the cheaper/faster model
    key="embedding_model_selector",
    help="`text-embedding-3-small` is faster and cheaper, while `text-embedding-3-large` offers potentially higher accuracy." # [1, 2, 3, 4, 5, 6, 7, 8]
)

# Placeholder for Number of Clusters (k) Input
num_clusters = st.sidebar.slider(
    "Select the number of clusters (k):",
    min_value=2,
    max_value=20, # Adjust max value as needed
    value=5, # Default value
    step=1,
    key="k_slider",
    help="Choose the desired number of keyword groups. Elbow/Silhouette plots (when generated) can help guide this selection." # [9, 10, 11, 12, 13, 14, 15]
)

# --- Button to Start Processing ---
st.sidebar.markdown("---")
start_processing = st.sidebar.button("Cluster Keywords", key="start_button", type="primary")

# --- Display Area ---
st.markdown("---")
st.header("Results")

if start_processing:
    if not api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar.")
    elif uploaded_file is None:
        st.warning("Please upload a CSV file containing keywords in the sidebar.")
    elif keyword_column is None:
         st.warning("Please select the keyword column in the sidebar.")
    else:
        st.info("Processing started... This may take a few moments depending on the file size and API speed.")
        # --- Placeholder for calling the core logic ---
        # 1. Read the full CSV using the selected keyword_column [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        # 2. Get embeddings using the selected model and API key [26, 1, 2, 3, 27, 4, 28, 5, 6, 7, 29, 8, 30, 31, 32, 33, 34, 35, 36, 37]
        # 3. Perform K-Means clustering [9, 38, 10, 11, 39, 40, 12, 13, 14, 15, 41, 42, 43]
        # 4. Determine optimal K (Elbow/Silhouette) [11, 12, 13, 14, 15]
        # 5. Visualize results (UMAP + Plotly) [44, 45, 46, 47, 48, 49, 50, 51, 52]
        # 6. Display clustered keywords table [45, 53]
        
        st.write(f"Processing file: {uploaded_file.name}")
        st.write(f"Selected keyword column: {keyword_column}")
        st.write(f"Selected embedding model: {embedding_model}")
        st.write(f"Selected number of clusters: {num_clusters}")
        
        # Example: Displaying a placeholder message
        st.success("Clustering process would run here!")
        
        # Placeholder for results display
        st.subheader("Cluster Visualization (Placeholder)")
        st.markdown("*(Interactive UMAP/t-SNE plot will appear here)*") # [47, 48, 49, 50, 52]
        
        st.subheader("Clustered Keywords (Placeholder)")
        st.markdown("*(Table showing keywords grouped by cluster will appear here)*")

else:
    st.info("Configure the options in the sidebar and click 'Cluster Keywords' to begin.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed for semantic keyword analysis.")
