import streamlit as st
import pandas as pd

from .utils import (
    generate_sample_csv,
    add_cost_calculator,
    show_csv_cost_estimate,
    validate_csv_content,
    sanitize_csv_data,
    SPACY_LANGUAGE_MODELS,
    LIBRARIES,
)
from .preprocessing import (
    download_nltk_resources,
    load_spacy_model_by_language,
    preprocess_keywords,
)
from .embeddings import generate_embeddings
from .clustering import improved_clustering_with_monitoring


def main():
    """
    Main function to render the Streamlit UI for semantic keyword clustering.
    """
    st.set_page_config(page_title="Semantic Keyword Clustering", layout="wide")
    st.title("Semantic Keyword Clustering Tool")

    # Ensure NLTK resources are downloaded
    download_nltk_resources()

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        # Sample CSV download
        if st.button("Download Sample CSV"):
            st.download_button(
                label="Download Example",
                data=generate_sample_csv(),
                file_name="sample_keywords.csv",
                mime="text/csv",
            )

        # CSV format option
        csv_format = st.selectbox(
            "CSV Format", ["Header row", "No header (keywords only)"], index=0
        )

        # File uploader
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"]);

        # OpenAI API key
        openai_key = st.text_input("OpenAI API Key", type="password")

        # Language selection
        language = st.selectbox(
            "Language",
            options=list(SPACY_LANGUAGE_MODELS.keys()),
            index=0
        )

        # Number of clusters
        num_clusters = st.number_input(
            "Number of clusters",
            min_value=2,
            max_value=50,
            value=10,
            step=1
        )

        add_cost_calculator()

    # Process uploaded file
    if 'uploaded_file' in locals() and uploaded_file is not None:
        # Read CSV into DataFrame
        header_row = 0 if csv_format == "Header row" else None
        df = pd.read_csv(uploaded_file, header=header_row)

        # Validate content
        valid, msg = validate_csv_content(df)
        if not valid:
            st.error(msg)
            return
        df = sanitize_csv_data(df)

        # Display cost estimate
        show_csv_cost_estimate(len(df), model="text-embedding-3-small", num_clusters=num_clusters)

        # Preprocess keywords
        st.subheader("Preprocessing Keywords")
        spacy_model = load_spacy_model_by_language(
            selected_language=language,
            model_map=SPACY_LANGUAGE_MODELS,
            base_available=LIBRARIES.get('spacy_base_available', False)
        )
        with st.spinner("Processing keywords..."):
            df['processed_keyword'] = preprocess_keywords(
                keywords=df.iloc[:, 0].tolist(),
                use_advanced=True,
                spacy_nlp=spacy_model,
                textblob_available=LIBRARIES.get('textblob_available', False)
            )

        # Generate embeddings
        st.subheader("Generating Embeddings")
        with st.spinner("Generating embeddings..."):
            embeddings = generate_embeddings(
                dataframe=df,
                text_column='processed_keyword',
                openai_api_key=openai_key
            )

        # Perform clustering
        st.subheader("Clustering Keywords")
        with st.spinner("Clustering..."):
            labels = improved_clustering_with_monitoring(embeddings, num_clusters)
            df['cluster_id'] = labels

        # Show results
        st.subheader("Clustered Keywords")
        st.dataframe(df)


if __name__ == "__main__":
    main()
