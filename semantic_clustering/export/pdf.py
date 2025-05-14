"""
PDF export functionality for semantic keyword clustering.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import tempfile
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Check if reportlab is installed
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Check if Pillow is installed
try:
    from PIL import Image as PILImage
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

def check_dependencies() -> bool:
    """
    Check if all required dependencies for PDF export are installed.
    
    Returns:
        True if all dependencies are installed, False otherwise
    """
    if not REPORTLAB_AVAILABLE:
        logger.error("ReportLab is required for PDF export. Install it using 'pip install reportlab'")
        return False
        
    if not PILLOW_AVAILABLE:
        logger.error("Pillow is required for PDF export. Install it using 'pip install pillow'")
        return False
        
    return True

def create_cluster_visualization(
    clusters: Dict[str, List[str]],
    embeddings_2d: np.ndarray = None,
    labels: List[int] = None,
    output_path: Optional[str] = None,
    width: int = 800,
    height: int = 600,
    dpi: int = 100,
    show_labels: bool = True
) -> Optional[str]:
    """
    Create a visualization of the clusters.
    
    Args:
        clusters: Dictionary of cluster_id -> list of keywords
        embeddings_2d: 2D embeddings for visualization (optional)
        labels: Cluster labels corresponding to embeddings_2d (optional)
        output_path: Path to save the visualization image (optional)
        width: Width of the visualization in pixels
        height: Height of the visualization in pixels
        dpi: Dots per inch for the output image
        show_labels: Whether to show cluster labels on the visualization
        
    Returns:
        Path to the saved visualization image, or None if visualization failed
    """
    if not embeddings_2d is not None and labels is not None:
        logger.warning("Both embeddings_2d and labels are required for visualization")
        return None
        
    if not PILLOW_AVAILABLE:
        logger.error("Pillow is required for visualization. Install it using 'pip install pillow'")
        return None
    
    # Create a temporary file if output_path is not provided
    if output_path is None:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name
    
    try:
        # Create plot
        fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
        
        # Get unique labels and assign colors
        unique_labels = np.unique(labels)
        colors_list = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each cluster with a different color
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                color=colors_list[i],
                label=f"Cluster {label}",
                alpha=0.7,
                s=50
            )
            
            # Add cluster label
            if show_labels:
                centroid = embeddings_2d[mask].mean(axis=0)
                ax.text(
                    centroid[0], 
                    centroid[1], 
                    f"Cluster {label}",
                    fontsize=12,
                    fontweight='bold',
                    ha='center', 
                    va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
                )
        
        ax.set_title("Keyword Clusters Visualization", fontsize=16)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating cluster visualization: {e}")
        return None

def create_cluster_report_pdf(
    clusters: Dict[str, List[str]],
    cluster_labels: Optional[Dict[str, str]] = None,
    evaluation_metrics: Optional[Dict[str, Any]] = None,
    visualization_path: Optional[str] = None,
    title: str = "Semantic Keyword Clustering Report",
    description: str = "",
    output_path: Optional[str] = None
) -> Optional[str]:
    """
    Create a PDF report for the clustering results.
    
    Args:
        clusters: Dictionary of cluster_id -> list of keywords
        cluster_labels: Dictionary of cluster_id -> descriptive label (optional)
        evaluation_metrics: Dictionary of evaluation metrics (optional)
        visualization_path: Path to the visualization image (optional)
        title: Title of the report
        description: Description of the clustering analysis
        output_path: Path to save the PDF report (optional)
        
    Returns:
        Path to the saved PDF report, or None if report generation failed
    """
    if not check_dependencies():
        return None
    
    # Create a temporary file if output_path is not provided
    if output_path is None:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            output_path = tmp.name
    
    try:
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading1_style = styles['Heading1']
        heading2_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # Create custom styles
        bullet_style = ParagraphStyle(
            'BulletPoint',
            parent=normal_style,
            leftIndent=20,
            firstLineIndent=0,
            spaceBefore=2,
            spaceAfter=2
        )
        
        # Build document content
        content = []
        
        # Title
        content.append(Paragraph(title, title_style))
        content.append(Spacer(1, 12))
        
        # Date
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content.append(Paragraph(f"Generated on: {date_str}", normal_style))
        content.append(Spacer(1, 12))
        
        # Description
        if description:
            content.append(Paragraph("Description:", heading2_style))
            content.append(Paragraph(description, normal_style))
            content.append(Spacer(1, 12))
        
        # Summary statistics
        content.append(Paragraph("Summary Statistics:", heading2_style))
        total_clusters = len(clusters)
        total_keywords = sum(len(keywords) for keywords in clusters.values())
        
        summary_data = [
            ["Total Clusters", str(total_clusters)],
            ["Total Keywords", str(total_keywords)],
            ["Average Keywords per Cluster", f"{total_keywords / total_clusters:.2f}" if total_clusters > 0 else "0"]
        ]
        
        # Add evaluation metrics to summary if provided
        if evaluation_metrics:
            for metric, value in evaluation_metrics.items():
                if isinstance(value, (int, float)):
                    summary_data.append([metric.replace("_", " ").title(), f"{value:.4f}"])
        
        summary_table = Table(summary_data, colWidths=[doc.width/2.5, doc.width/2.5])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT')
        ]))
        
        content.append(summary_table)
        content.append(Spacer(1, 12))
        
        # Visualization if provided
        if visualization_path and os.path.exists(visualization_path):
            content.append(Paragraph("Cluster Visualization:", heading2_style))
            img_width = doc.width
            img_height = img_width * 0.75  # Maintain aspect ratio
            content.append(Image(visualization_path, width=img_width, height=img_height))
            content.append(Spacer(1, 12))
        
        # Clusters details
        content.append(Paragraph("Cluster Details:", heading2_style))
        content.append(Spacer(1, 6))
        
        # Sort clusters by size (descending)
        sorted_clusters = sorted(
            clusters.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for i, (cluster_id, keywords) in enumerate(sorted_clusters):
            # Add page break every 3 clusters
            if i > 0 and i % 3 == 0:
                content.append(PageBreak())
            
            # Cluster header
            if cluster_labels and cluster_id in cluster_labels:
                label = cluster_labels[cluster_id]
                content.append(Paragraph(f"Cluster {cluster_id}: {label}", heading2_style))
            else:
                content.append(Paragraph(f"Cluster {cluster_id}", heading2_style))
            
            # Keywords
            content.append(Paragraph(f"Size: {len(keywords)} keywords", normal_style))
            content.append(Spacer(1, 6))
            
            # Display keywords in a table with multiple columns
            num_cols = 3
            keywords_sorted = sorted(keywords)
            num_rows = (len(keywords_sorted) + num_cols - 1) // num_cols
            
            # Fill the table data
            table_data = []
            for row in range(num_rows):
                table_row = []
                for col in range(num_cols):
                    idx = row + col * num_rows
                    if idx < len(keywords_sorted):
                        table_row.append(keywords_sorted[idx])
                    else:
                        table_row.append("")
                table_data.append(table_row)
            
            # Create table
            keyword_table = Table(table_data, colWidths=[doc.width/num_cols] * num_cols)
            keyword_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('PADDING', (0, 0), (-1, -1), 4),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
            ]))
            
            content.append(keyword_table)
            content.append(Spacer(1, 12))
        
        # Build the PDF
        doc.build(content)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating PDF report: {e}")
        return None

def export_to_pdf(
    clusters: Dict[str, List[str]],
    output_path: str,
    embeddings_2d: Optional[np.ndarray] = None,
    labels: Optional[List[int]] = None,
    cluster_labels: Optional[Dict[str, str]] = None,
    evaluation_metrics: Optional[Dict[str, Any]] = None,
    title: str = "Semantic Keyword Clustering Report",
    description: str = ""
) -> bool:
    """
    Export clustering results to PDF.
    
    Args:
        clusters: Dictionary of cluster_id -> list of keywords
        output_path: Path to save the PDF report
        embeddings_2d: 2D embeddings for visualization (optional)
        labels: Cluster labels corresponding to embeddings_2d (optional)
        cluster_labels: Dictionary of cluster_id -> descriptive label (optional)
        evaluation_metrics: Dictionary of evaluation metrics (optional)
        title: Title of the report
        description: Description of the clustering analysis
        
    Returns:
        True if PDF export was successful, False otherwise
    """
    if not check_dependencies():
        return False
    
    try:
        # Create visualization if embeddings and labels are provided
        visualization_path = None
        if embeddings_2d is not None and labels is not None:
            visualization_path = create_cluster_visualization(
                clusters,
                embeddings_2d=embeddings_2d,
                labels=labels
            )
        
        # Create PDF report
        pdf_path = create_cluster_report_pdf(
            clusters,
            cluster_labels=cluster_labels,
            evaluation_metrics=evaluation_metrics,
            visualization_path=visualization_path,
            title=title,
            description=description,
            output_path=output_path
        )
        
        # Clean up temporary visualization file
        if visualization_path is not None and os.path.exists(visualization_path) and visualization_path != output_path:
            os.remove(visualization_path)
        
        return pdf_path is not None and os.path.exists(pdf_path)
        
    except Exception as e:
        logger.error(f"Error exporting to PDF: {e}")
        return False
