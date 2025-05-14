"""
JSON export functionality for semantic keyword clustering.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import json
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

def format_clusters_for_json(
    clusters: Dict[str, List[str]],
    cluster_labels: Optional[Dict[str, str]] = None,
    evaluation_metrics: Optional[Dict[str, Any]] = None,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Format clustering results for JSON export.
    
    Args:
        clusters: Dictionary of cluster_id -> list of keywords
        cluster_labels: Dictionary of cluster_id -> descriptive label (optional)
        evaluation_metrics: Dictionary of evaluation metrics (optional)
        include_metadata: Whether to include metadata in the output
        
    Returns:
        Dictionary formatted for JSON export
    """
    # Initialize result dictionary
    result = {
        "clusters": []
    }
    
    # Add metadata if requested
    if include_metadata:
        result["metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "total_clusters": len(clusters),
            "total_keywords": sum(len(keywords) for keywords in clusters.values())
        }
        
        # Add evaluation metrics if provided
        if evaluation_metrics:
            # Filter out complex objects that can't be serialized to JSON
            metrics = {}
            for k, v in evaluation_metrics.items():
                if isinstance(v, (int, float, str, bool, list, dict)) and not isinstance(v, np.ndarray):
                    # Convert float values to fixed precision strings
                    if isinstance(v, float):
                        metrics[k] = round(v, 6)
                    else:
                        metrics[k] = v
            
            result["metadata"]["evaluation_metrics"] = metrics
    
    # Sort clusters by size (descending)
    sorted_clusters = sorted(
        clusters.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    # Add clusters
    for cluster_id, keywords in sorted_clusters:
        cluster_data = {
            "id": cluster_id,
            "size": len(keywords),
            "keywords": sorted(keywords)
        }
        
        # Add label if available
        if cluster_labels and cluster_id in cluster_labels:
            cluster_data["label"] = cluster_labels[cluster_id]
        
        result["clusters"].append(cluster_data)
    
    return result

def export_to_json(
    clusters: Dict[str, List[str]],
    output_path: str,
    cluster_labels: Optional[Dict[str, str]] = None,
    evaluation_metrics: Optional[Dict[str, Any]] = None,
    include_metadata: bool = True,
    pretty: bool = True
) -> bool:
    """
    Export clustering results to JSON.
    
    Args:
        clusters: Dictionary of cluster_id -> list of keywords
        output_path: Path to save the JSON file
        cluster_labels: Dictionary of cluster_id -> descriptive label (optional)
        evaluation_metrics: Dictionary of evaluation metrics (optional)
        include_metadata: Whether to include metadata in the output
        pretty: Whether to format the JSON with indentation for readability
        
    Returns:
        True if JSON export was successful, False otherwise
    """
    try:
        # Format data for JSON
        data = format_clusters_for_json(
            clusters,
            cluster_labels=cluster_labels,
            evaluation_metrics=evaluation_metrics,
            include_metadata=include_metadata
        )
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        logger.error(f"Error exporting to JSON: {e}")
        return False

def export_cluster_as_jsonl(
    clusters: Dict[str, List[str]],
    output_path: str,
    cluster_labels: Optional[Dict[str, str]] = None
) -> bool:
    """
    Export clusters as JSONL (JSON Lines) format, with one keyword per line.
    
    This format is useful for bulk importing into other systems or for
    processing with streaming JSON parsers.
    
    Args:
        clusters: Dictionary of cluster_id -> list of keywords
        output_path: Path to save the JSONL file
        cluster_labels: Dictionary of cluster_id -> descriptive label (optional)
        
    Returns:
        True if JSONL export was successful, False otherwise
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for cluster_id, keywords in clusters.items():
                label = cluster_labels.get(cluster_id, "") if cluster_labels else ""
                
                for keyword in keywords:
                    item = {
                        "cluster_id": cluster_id,
                        "keyword": keyword
                    }
                    
                    if label:
                        item["cluster_label"] = label
                    
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return True
        
    except Exception as e:
        logger.error(f"Error exporting to JSONL: {e}")
        return False

def export_jsonl_for_elasticsearch(
    clusters: Dict[str, List[str]],
    output_path: str,
    index_name: str = "keywords",
    cluster_labels: Optional[Dict[str, str]] = None
) -> bool:
    """
    Export clusters as Elasticsearch bulk import format.
    
    This creates a JSONL file with action/metadata and document pairs
    that can be directly imported into Elasticsearch.
    
    Args:
        clusters: Dictionary of cluster_id -> list of keywords
        output_path: Path to save the Elasticsearch bulk import file
        index_name: Name of the Elasticsearch index
        cluster_labels: Dictionary of cluster_id -> descriptive label (optional)
        
    Returns:
        True if Elasticsearch export was successful, False otherwise
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for cluster_id, keywords in clusters.items():
                label = cluster_labels.get(cluster_id, "") if cluster_labels else ""
                
                for i, keyword in enumerate(keywords):
                    # Add action/metadata line
                    action = {
                        "index": {
                            "_index": index_name,
                            "_id": f"{cluster_id}-{i}"
                        }
                    }
                    f.write(json.dumps(action, ensure_ascii=False) + '\n')
                    
                    # Add document line
                    doc = {
                        "keyword": keyword,
                        "cluster_id": cluster_id,
                        "cluster_size": len(keywords)
                    }
                    
                    if label:
                        doc["cluster_label"] = label
                    
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        return True
        
    except Exception as e:
        logger.error(f"Error exporting for Elasticsearch: {e}")
        return False

def import_clusters_from_json(
    input_path: str
) -> Tuple[Dict[str, List[str]], Optional[Dict[str, str]], Optional[Dict[str, Any]]]:
    """
    Import clusters from a JSON file.
    
    Args:
        input_path: Path to the JSON file
        
    Returns:
        Tuple of (clusters dict, cluster labels dict, evaluation metrics dict)
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Initialize result
        clusters = {}
        cluster_labels = {}
        evaluation_metrics = None
        
        # Extract clusters
        for cluster_data in data.get("clusters", []):
            cluster_id = str(cluster_data.get("id", ""))
            keywords = cluster_data.get("keywords", [])
            
            if cluster_id and keywords:
                clusters[cluster_id] = keywords
                
                # Extract label if available
                if "label" in cluster_data:
                    cluster_labels[cluster_id] = cluster_data["label"]
        
        # Extract evaluation metrics if available
        if "metadata" in data and "evaluation_metrics" in data["metadata"]:
            evaluation_metrics = data["metadata"]["evaluation_metrics"]
        
        return clusters, cluster_labels, evaluation_metrics
        
    except Exception as e:
        logger.error(f"Error importing from JSON: {e}")
        return {}, None, None
