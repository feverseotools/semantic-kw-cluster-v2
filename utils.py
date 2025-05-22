import os
import logging
import psutil
import streamlit as st
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)

# Configuration constants (override via environment variables)
OPENAI_TIMEOUT = float(os.getenv('OPENAI_TIMEOUT', '60.0'))
OPENAI_MAX_RETRIES = int(os.getenv('OPENAI_MAX_RETRIES', '3'))
MAX_KEYWORDS = int(os.getenv('MAX_KEYWORDS', '25000'))
MAX_MEMORY_WARNING_MB = int(os.getenv('MAX_MEMORY_MB', '800'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '100'))

# These can be populated in app startup based on availability
SPACY_LANGUAGE_MODELS = {}
LIBRARIES = {}


def monitor_resources():
    """
    Monitor memory usage of the current process and warn if exceeding threshold.
    Updates session_state with peak memory usage.
    """
    try:
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        if mem_mb > MAX_MEMORY_WARNING_MB:
            st.warning(f"High memory usage detected: {mem_mb:.1f} MB")
        peak = st.session_state.setdefault('memory_monitor', {}).get('peak_memory', 0)
        st.session_state['memory_monitor']['peak_memory'] = max(peak, mem_mb)
    except Exception as e:
        logger.error(f"Resource monitoring failed: {e}")


def generate_sample_csv():
    """
    Generate a sample CSV string for keywords, volumes, and CPC data.
    Returns:
        str: CSV-formatted text.
    """
    header = ["Keyword", "search_volume", "competition", "cpc"] + [f"month_{i}" for i in range(1, 13)]
    rows = [
        ["running shoes", 5400, 0.75, 1.25] + list(range(450, 560, 10)),
        ["smart watch", 3200, 0.60, 2.10] + list(range(300, 420, 10)),
        ["wireless earbuds", 4100, 0.80, 1.95] + list(range(380, 480, 10)),
    ]
    import io, csv
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def calculate_api_cost(num_keywords: int, model: str, num_clusters: int, cost_per_1k_tokens: float = 0.0004) -> float:
    """
    Estimate OpenAI API cost based on number of keywords and model.

    Args:
        num_keywords (int): Number of keywords to embed.
        model (str): Model name (unused placeholder).
        num_clusters (int): Number of clusters (unused placeholder).
        cost_per_1k_tokens (float): Cost per 1k tokens in USD.

    Returns:
        float: Estimated cost in USD.
    """
    # Rough estimate: assume 1 keyword ~ 1 token; adjust as needed
    tokens = num_keywords
n    cost = tokens / 1000 * cost_per_1k_tokens
    return round(cost, 4)


def add_cost_calculator():
    """
    Add a cost estimate section to the Streamlit sidebar.
    """
    st.sidebar.markdown("---")
    st.sidebar.write("Cost estimator: function calculate_api_cost(num_keywords, model, clusters)")


def show_csv_cost_estimate(num_rows: int, model: str, num_clusters: int):
    """
    Display API cost estimate for the uploaded CSV.

    Args:
        num_rows (int): Number of rows in the CSV.
        model (str): Model name.
        num_clusters (int): Number of clusters.
    """
    cost = calculate_api_cost(num_rows, model, num_clusters)
    st.sidebar.info(f"Estimated API cost: ${cost}")


def sanitize_text(text: str) -> str:
    """
    Remove HTML tags and ensure text is a string.

    Args:
        text (Any): Input text.

    Returns:
        str: Cleaned text.
    """
    return re.sub(r'<[^>]+>', '', str(text))


def validate_csv_content(df) -> (bool, str):
    """
    Basic validation for uploaded CSV DataFrame.

    Checks for empty DataFrame.

    Returns:
        (bool, str): Tuple indicating validity and message.
    """
    if df.empty:
        return False, "Uploaded CSV is empty."
    # Further schema checks can be added here
    return True, "CSV content is valid."


def sanitize_csv_data(df):
    """
    Sanitize DataFrame content: drop NaNs and reset index.

    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    df_clean = df_clean.dropna().reset_index(drop=True)
    return df_clean
