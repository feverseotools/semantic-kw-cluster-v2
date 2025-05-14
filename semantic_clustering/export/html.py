"""
HTML export functionality for semantic keyword clustering.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import tempfile
from datetime import datetime
import json
import base64
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

logger = logging.getLogger(__name__)

def get_cluster_colors(n_clusters: int) -> List[str]:
    """
    Generate a list of distinct colors for clusters.
    
    Args:
        n_clusters: Number of colors to generate
        
    Returns:
        List of hex color codes
    """
    cmap = plt.cm.get_cmap('tab20', n_clusters)
    colors = []
    
    for i in range(n_clusters):
        rgba = cmap(i)
        # Convert to hex
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgba[0] * 255),
            int(rgba[1] * 255),
            int(rgba[2] * 255)
        )
        colors.append(hex_color)
    
    return colors

def create_cluster_visualization_base64(
    embeddings_2d: np.ndarray,
    labels: List[int],
    cluster_labels: Optional[Dict[str, str]] = None,
    width: int = 800,
    height: int = 600,
    dpi: int = 100,
    show_legend: bool = True
) -> str:
    """
    Create a base64-encoded visualization of clusters.
    
    Args:
        embeddings_2d: 2D embeddings for visualization
        labels: Cluster labels corresponding to embeddings_2d
        cluster_labels: Dictionary of cluster_id -> descriptive label (optional)
        width: Width of the visualization in pixels
        height: Height of the visualization in pixels
        dpi: Dots per inch for the output image
        show_legend: Whether to show a legend
        
    Returns:
        Base64-encoded PNG image
    """
    try:
        # Create figure
        fig = Figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        ax = fig.add_subplot(111)
        
        # Get unique labels and colors
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        colors = get_cluster_colors(n_clusters)
        
        # Plot each cluster
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            color = colors[i % len(colors)]
            
            # Get label text
            if cluster_labels and str(label) in cluster_labels:
                label_text = f"Cluster {label}: {cluster_labels[str(label)]}"
            else:
                label_text = f"Cluster {label}"
            
            # Plot points
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color],
                label=label_text,
                alpha=0.7,
                s=50,
                edgecolors='w',
                linewidths=0.5
            )
            
            # Add cluster label at centroid
            if len(embeddings_2d[mask]) > 0:
                centroid = embeddings_2d[mask].mean(axis=0)
                ax.text(
                    centroid[0],
                    centroid[1],
                    str(label),
                    fontsize=12,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
                )
        
        # Set plot properties
        ax.set_title("Keyword Clusters Visualization", fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend if requested
        if show_legend and n_clusters <= 20:  # Only show legend if not too many clusters
            ax.legend(
                loc='best',
                fontsize=10,
                markerscale=0.7,
                frameon=True,
                framealpha=0.8
            )
        
        # Remove axes labels and ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save to base64
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        
        data = base64.b64encode(buf.read()).decode('ascii')
        plt.close(fig)
        
        return f"data:image/png;base64,{data}"
        
    except Exception as e:
        logger.error(f"Error creating cluster visualization: {e}")
        return ""

def create_cluster_size_chart_base64(
    clusters: Dict[str, List[str]],
    width: int = 800,
    height: int = 500,
    dpi: int = 100
) -> str:
    """
    Create a base64-encoded bar chart of cluster sizes.
    
    Args:
        clusters: Dictionary of cluster_id -> list of keywords
        width: Width of the chart in pixels
        height: Height of the chart in pixels
        dpi: Dots per inch for the output image
        
    Returns:
        Base64-encoded PNG image
    """
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
        fig = Figure(figsize=(width/dpi, height/dpi), dpi=dpi)
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
        
        # Save to base64
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        
        data = base64.b64encode(buf.read()).decode('ascii')
        plt.close(fig)
        
        return f"data:image/png;base64,{data}"
        
    except Exception as e:
        logger.error(f"Error creating cluster size chart: {e}")
        return ""

def generate_html_report(
    clusters: Dict[str, List[str]],
    cluster_labels: Optional[Dict[str, str]] = None,
    evaluation_metrics: Optional[Dict[str, Any]] = None,
    embeddings_2d: Optional[np.ndarray] = None,
    labels: Optional[List[int]] = None,
    title: str = "Semantic Keyword Clustering Report",
    description: str = ""
) -> str:
    """
    Generate an HTML report for clustering results.
    
    Args:
        clusters: Dictionary of cluster_id -> list of keywords
        cluster_labels: Dictionary of cluster_id -> descriptive label (optional)
        evaluation_metrics: Dictionary of evaluation metrics (optional)
        embeddings_2d: 2D embeddings for visualization (optional)
        labels: Cluster labels corresponding to embeddings_2d (optional)
        title: Title of the report
        description: Description of the clustering analysis
        
    Returns:
        HTML string
    """
    try:
        # Start HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: #fff;
            padding: 30px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .cluster {{
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
            border-left: 5px solid #3498db;
        }}
        .cluster-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }}
        .cluster-title {{
            font-weight: bold;
            font-size: 1.2em;
            color: #2c3e50;
        }}
        .cluster-size {{
            color: #7f8c8d;
        }}
        .keywords {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .keyword {{
            background-color: #e1f0fa;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }}
        .metric-card {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .metric-name {{
            font-weight: bold;
            margin-bottom: 5px;
            color: #2c3e50;
        }}
        .metric-value {{
            font-size: 1.2em;
            color: #3498db;
        }}
        .visualization {{
            text-align: center;
            margin: 20px 0;
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .summary {{
            background-color: #f0f7fb;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 5px solid #3498db;
        }}
        .footer {{
            margin-top: 50px;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.8em;
        }}
        .tabs {{
            display: flex;
            border-bottom: 1px solid #ddd;
            margin-bottom: 20px;
        }}
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            background-color: #f1f1f1;
            margin-right: 5px;
            border-radius: 5px 5px 0 0;
        }}
        .tab.active {{
            background-color: #3498db;
            color: white;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div class="summary">
            <p><strong>Generated on:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Total clusters:</strong> {len(clusters)}</p>
            <p><strong>Total keywords:</strong> {sum(len(keywords) for keywords in clusters.values())}</p>
        """
        
        # Add description if provided
        if description:
            html += f"<p><strong>Description:</strong> {description}</p>\n"
            
        html += "</div>\n"  # Close summary div
        
        # Create tabs
        html += """
        <div class="tabs">
            <div class="tab active" onclick="openTab(event, 'overview')">Overview</div>
            <div class="tab" onclick="openTab(event, 'clusters')">Clusters</div>
            <div class="tab" onclick="openTab(event, 'metrics')">Metrics</div>
        </div>
        """
        
        # Overview tab
        html += '<div id="overview" class="tab-content active">\n'
        
        # Add cluster size visualization
        cluster_size_chart = create_cluster_size_chart_base64(clusters)
        if cluster_size_chart:
            html += f"""
            <div class="section">
                <h2>Cluster Sizes</h2>
                <div class="visualization">
                    <img src="{cluster_size_chart}" alt="Cluster Sizes">
                </div>
            </div>
            """
        
        # Add cluster visualization if embeddings are provided
        if embeddings_2d is not None and labels is not None:
            visualization = create_cluster_visualization_base64(embeddings_2d, labels, cluster_labels)
            if visualization:
                html += f"""
                <div class="section">
                    <h2>Cluster Visualization</h2>
                    <div class="visualization">
                        <img src="{visualization}" alt="Cluster Visualization">
                    </div>
                </div>
                """
        
        html += '</div>\n'  # Close overview tab
        
        # Clusters tab
        html += '<div id="clusters" class="tab-content">\n'
        html += '<div class="section">\n'
        html += '<h2>Cluster Details</h2>\n'
        
        # Sort clusters by size (descending)
        sorted_clusters = sorted(
            clusters.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        # Add cluster details
        for cluster_id, keywords in sorted_clusters:
            # Get cluster label
            label = cluster_labels.get(cluster_id, "") if cluster_labels else ""
            
            # Generate cluster HTML
            html += f"""
            <div class="cluster">
                <div class="cluster-header">
                    <div class="cluster-title">Cluster {cluster_id}{f": {label}" if label else ""}</div>
                    <div class="cluster-size">{len(keywords)} keywords</div>
                </div>
                <div class="keywords">
            """
            
            # Add keywords
            for keyword in sorted(keywords):
                html += f'<div class="keyword">{keyword}</div>\n'
            
            html += "</div>\n</div>\n"  # Close keywords and cluster divs
        
        html += '</div>\n'  # Close section div
        html += '</div>\n'  # Close clusters tab
        
        # Metrics tab
        html += '<div id="metrics" class="tab-content">\n'
        html += '<div class="section">\n'
        html += '<h2>Evaluation Metrics</h2>\n'
        
        if evaluation_metrics:
            html += '<div class="metrics">\n'
            
            # Add key metrics in cards
            for metric, value in evaluation_metrics.items():
                # Skip complex data structures
                if not isinstance(value, (int, float, str, bool)):
                    continue
                
                # Format metric name and value
                metric_name = metric.replace("_", " ").title()
                
                if isinstance(value, float):
                    metric_value = f"{value:.4f}"
                else:
                    metric_value = str(value)
                
                html += f"""
                <div class="metric-card">
                    <div class="metric-name">{metric_name}</div>
                    <div class="metric-value">{metric_value}</div>
                </div>
                """
            
            html += '</div>\n'  # Close metrics div
        else:
            html += '<p>No evaluation metrics available.</p>\n'
        
        html += '</div>\n'  # Close section div
        html += '</div>\n'  # Close metrics tab
        
        # Add JavaScript for tabs
        html += """
        <script>
            function openTab(evt, tabName) {
                // Hide all tab content
                var tabContents = document.getElementsByClassName("tab-content");
                for (var i = 0; i < tabContents.length; i++) {
                    tabContents[i].className = tabContents[i].className.replace(" active", "");
                }
                
                // Remove active class from all tabs
                var tabs = document.getElementsByClassName("tab");
                for (var i = 0; i < tabs.length; i++) {
                    tabs[i].className = tabs[i].className.replace(" active", "");
                }
                
                // Show the selected tab content and add active class to the button
                document.getElementById(tabName).className += " active";
                evt.currentTarget.className += " active";
            }
        </script>
        """
        
        # Add footer
        html += f"""
        <div class="footer">
            <p>Generated by Semantic Keyword Clustering tool</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html
        
    except Exception as e:
        logger.error(f"Error generating HTML report: {e}")
        return f"<html><body><h1>Error generating report</h1><p>{str(e)}</p></body></html>"

def export_to_html(
    clusters: Dict[str, List[str]],
    output_path: str,
    cluster_labels: Optional[Dict[str, str]] = None,
    evaluation_metrics: Optional[Dict[str, Any]] = None,
    embeddings_2d: Optional[np.ndarray] = None,
    labels: Optional[List[int]] = None,
    title: str = "Semantic Keyword Clustering Report",
    description: str = ""
) -> bool:
    """
    Export clustering results to HTML.
    
    Args:
        clusters: Dictionary of cluster_id -> list of keywords
        output_path: Path to save the HTML file
        cluster_labels: Dictionary of cluster_id -> descriptive label (optional)
        evaluation_metrics: Dictionary of evaluation metrics (optional)
        embeddings_2d: 2D embeddings for visualization (optional)
        labels: Cluster labels corresponding to embeddings_2d (optional)
        title: Title of the report
        description: Description of the clustering analysis
        
    Returns:
        True if HTML export was successful, False otherwise
    """
    try:
        # Generate HTML report
        html = generate_html_report(
            clusters,
            cluster_labels=cluster_labels,
            evaluation_metrics=evaluation_metrics,
            embeddings_2d=embeddings_2d,
            labels=labels,
            title=title,
            description=description
        )
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return True
        
    except Exception as e:
        logger.error(f"Error exporting to HTML: {e}")
        return False
