Semantic Keyword Clustering - Usage Guide
Table of Contents

Introduction
Installation
Basic Usage

Command Line Interface
Python API


Input Formats
Clustering Options

Clustering Methods
Number of Clusters
Embedding Models


Output Formats

JSON
Excel
HTML
PDF


Advanced Options

Preprocessing Keywords
Optimizing Cluster Count
Cluster Labeling Methods


Examples
Troubleshooting

Introduction
Semantic Keyword Clustering is a powerful tool for organizing keywords based on their semantic meaning. Unlike traditional keyword clustering tools that rely on lexical similarity, this tool uses advanced embedding models to understand the contextual meaning of keywords, resulting in more intuitive and useful clusters.
This guide covers how to use the tool, from basic command-line operations to advanced API integration.
Installation
To install the package, use pip:
bash# Basic installation
pip install semantic-keyword-clustering

# Install with all optional dependencies
pip install semantic-keyword-clustering[full]

# Install with specific components
pip install semantic-keyword-clustering[export]  # Install with export capabilities
After installation, you may need to download required NLTK data:
bashpython -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
Basic Usage
Command Line Interface
The simplest way to use the tool is through the command line interface:
bash# Basic usage
semantic-clustering --input keywords.csv --output-dir results

# Specify clustering method and number of clusters
semantic-clustering --input keywords.csv --method kmeans --clusters 10

# Export to multiple formats
semantic-clustering --input keywords.csv --formats json excel html pdf
Python API
For more programmatic control, use the Python API:
pythonfrom semantic_clustering.app import SemanticKeywordClusterer

# Initialize the clusterer
clusterer = SemanticKeywordClusterer(
    embedding_model="all-MiniLM-L6-v2",
    method="kmeans",
    n_clusters=10
)

# Load keywords from a file
clusterer.load_keywords_from_file("keywords.csv")

# Or load keywords from a list
keywords = ["digital marketing", "social media", "seo strategy", ...]
clusterer.load_keywords(keywords)

# Perform clustering
clusters = clusterer.cluster()

# Save results
output_files = clusterer.save(
    output_dir="results",
    formats=["json", "excel", "html"],
    file_prefix="my_clusters"
)

# Access clustering results directly
metrics = clusterer.get_metrics()
cluster_labels = clusterer.get_cluster_labels()
embeddings_2d, labels = clusterer.get_visualization_data()
Input Formats
The tool supports several input formats:

CSV: First column should contain keywords. Additional columns (like search volume) are ignored for clustering but are preserved in the output.
TXT: One keyword per line.
JSON: Either an array of keywords or an object with a keywords property.

Example CSV:
keyword,search_volume,competition
digital marketing strategy,5400,0.75
seo best practices,3600,0.82
...
Example TXT:
digital marketing strategy
seo best practices
...
Example JSON:
json{
  "keywords": [
    "digital marketing strategy",
    "seo best practices",
    ...
  ]
}
Clustering Options
Clustering Methods
The tool supports four clustering methods:

KMeans (kmeans): Fast and works well for most cases. Produces equally sized clusters. Default method.
DBSCAN (dbscan): Density-based clustering that can identify outliers. Good for finding natural groupings.
HDBSCAN (hdbscan): Enhanced version of DBSCAN that adapts better to varying densities.
Agglomerative (agglomerative): Hierarchical clustering that builds nested clusters.

Example:
bashsemantic-clustering --input keywords.csv --method hdbscan
Number of Clusters
You can specify the number of clusters explicitly (required for KMeans and Agglomerative):
bashsemantic-clustering --input keywords.csv --clusters 15
For DBSCAN and HDBSCAN, the --clusters parameter is ignored as these methods automatically determine the number of clusters.
Embedding Models
The tool uses sentence-transformers models for generating keyword embeddings. Available models include:

all-MiniLM-L6-v2 (default): Fast and efficient model with good performance
all-mpnet-base-v2: Higher quality but slower
paraphrase-multilingual-MiniLM-L12-v2: Good for multilingual keywords

To list all available models:
bashsemantic-clustering --list-models
Example:
bashsemantic-clustering --input keywords.csv --embedding-model all-mpnet-base-v2
Output Formats
JSON
The JSON output is a structured representation of clusters with metadata:
json{
  "metadata": {
    "generated_at": "2023-05-14T12:34:56.789012",
    "total_clusters": 10,
    "total_keywords": 150,
    "evaluation_metrics": {
      "silhouette_score": 0.6543,
      ...
    }
  },
  "clusters": [
    {
      "id": "0",
      "label": "seo content strategy",
      "size": 15,
      "keywords": ["seo strategy", "content marketing", ...]
    },
    ...
  ]
}
Excel
The Excel output contains multiple sheets:

Summary: Overview with key metrics
Clusters: Detailed listing of all clusters and their keywords
Metrics: Complete evaluation metrics (if available)

HTML
The HTML output is an interactive report with:

Visualizations of cluster sizes
2D visualization of keyword embeddings (if available)
Interactive tabs for Overview, Clusters, and Metrics
Responsive design for viewing on different devices

PDF
The PDF output is a printable report with:

Cluster visualizations
Summary statistics
Detailed listing of all clusters and their keywords
Evaluation metrics

Advanced Options
Preprocessing Keywords
By default, keywords are preprocessed to remove punctuation, convert to lowercase, etc. To disable preprocessing:
bashsemantic-clustering --input keywords.csv --no-preprocess
In the Python API:
pythonclusterer = SemanticKeywordClusterer(perform_preprocessing=False)
Optimizing Cluster Count
The tool can automatically determine the optimal number of clusters:
bashsemantic-clustering --input keywords.csv --optimize --min-clusters 5 --max-clusters 30
This uses silhouette analysis to find the optimal number of clusters within the specified range.
Cluster Labeling Methods
The tool has several methods for generating descriptive cluster labels:

TFIDF (tfidf): Uses TF-IDF to find the most distinctive terms in each cluster. Default method.
Frequent (frequent): Uses the most frequently occurring terms in each cluster.
Centroid (centroid): Uses keywords closest to the cluster centroid.

Examples
Basic Clustering with Default Settings
bashsemantic-clustering --input data/keywords.csv --output-dir results
Optimized Clustering with Multiple Exports
bashsemantic-clustering --input data/keywords.csv --optimize --method kmeans --formats json html pdf --output-dir results
Advanced Clustering with Custom Settings
bashsemantic-clustering --input data/keywords.csv --method hdbscan --embedding-model all-mpnet-base-v2 --formats json excel --prefix industry_clusters --output-dir results
Troubleshooting
Missing Dependencies
If you encounter errors about missing dependencies, install the required packages:
bash# For PDF export
pip install reportlab pillow

# For Excel export
pip install openpyxl

# For advanced clustering methods
pip install hdbscan umap-learn
Memory Issues
If you encounter memory issues with large keyword sets:

Use a smaller embedding model (e.g., all-MiniLM-L6-v2 instead of all-mpnet-base-v2)
Process your keywords in batches
Increase your system's available memory

Clustering Quality Issues
If the clustering results are not satisfactory:

Try different clustering methods (e.g., hdbscan for more natural groupings)
Adjust the number of clusters or optimize the cluster count
Use a more powerful embedding model for better semantic understanding
Make sure your keywords are related enough to form meaningful clusters

For more detailed troubleshooting, refer to the logs which can be enabled with the --debug flag.
