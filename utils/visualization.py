import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

@st.cache_data
def create_cluster_visualization(df):
    """
    Create a bar chart visualization of cluster sizes.
    
    Args:
        df (pandas.DataFrame): DataFrame with clustering results
        
    Returns:
        plotly.graph_objects.Figure: Bar chart of cluster sizes
    """
    # Get cluster sizes
    cluster_sizes = df.groupby(['cluster_id', 'cluster_name']).size().reset_index(name='count')
    
    # Sort by size
    cluster_sizes = cluster_sizes.sort_values('count', ascending=False)
    
    # Create labels that include both name and size
    cluster_sizes['label'] = cluster_sizes.apply(
        lambda x: f"{x['cluster_name']} ({x['count']})", axis=1
    )
    
    # Create bar chart
    fig = px.bar(
        cluster_sizes,
        x='label',
        y='count',
        color='count',
        labels={'count': 'Number of Keywords', 'label': 'Cluster'},
        title='Cluster Size Distribution',
        color_continuous_scale='Viridis'
    )
    
    # Improve layout
    fig.update_layout(
        xaxis_title="Cluster",
        yaxis_title="Number of Keywords",
        xaxis_tickangle=-45,
        height=500,
        margin=dict(l=20, r=20, t=40, b=100)
    )
    
    return fig

@st.cache_data
def create_intent_visualization(cluster_intents):
    """
    Create a pie chart visualization of search intent distribution.
    
    Args:
        cluster_intents (dict): Dictionary of cluster intent analysis results
        
    Returns:
        plotly.graph_objects.Figure: Pie chart of intent distribution
    """
    # Count primary intents across all clusters
    intent_counts = {}
    for cluster_id, data in cluster_intents.items():
        primary_intent = data['intent']['primary_intent']
        intent_counts[primary_intent] = intent_counts.get(primary_intent, 0) + 1
    
    # Create dataframe for visualization
    intent_df = pd.DataFrame({
        'intent': list(intent_counts.keys()),
        'count': list(intent_counts.values())
    })
    
    # Create pie chart
    fig = px.pie(
        intent_df,
        names='intent',
        values='count',
        title='Search Intent Distribution',
        color='intent',
        color_discrete_map={
            'Informational': '#2196F3',
            'Commercial': '#9C27B0',
            'Transactional': '#FF9800',
            'Navigational': '#4CAF50',
            'Mixed Intent': '#9E9E9E'
        }
    )
    
    # Improve layout
    fig.update_layout(
        legend_title="Search Intent",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

@st.cache_data
def create_journey_visualization(cluster_intents):
    """
    Create a visualization of customer journey phases.
    
    Args:
        cluster_intents (dict): Dictionary of cluster intent analysis results
        
    Returns:
        plotly.graph_objects.Figure: Bar chart of journey phase distribution
    """
    # Extract journey phases
    journey_phases = {}
    for cluster_id, data in cluster_intents.items():
        journey_phase = data.get('journey_phase', 'Mixed')
        journey_phases[cluster_id] = journey_phase
    
    # Count phases
    phase_counts = {}
    for phase in journey_phases.values():
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    # Create dataframe
    phases_df = pd.DataFrame({
        'phase': list(phase_counts.keys()),
        'count': list(phase_counts.values())
    })
    
    # Define preferred order of phases
    phase_order = [
        "Early (Research/Awareness)",
        "Middle (Consideration)",
        "Late (Decision/Purchase)",
        "Mixed"
    ]
    
    # Order phases in the dataframe
    phases_df['order'] = phases_df['phase'].apply(
        lambda x: phase_order.index(x) if x in phase_order else len(phase_order)
    )
    phases_df = phases_df.sort_values('order')
    phases_df = phases_df.drop('order', axis=1)
    
    # Create bar chart
    fig = px.bar(
        phases_df, 
        x='phase', 
        y='count',
        color='phase',
        title="Customer Journey Distribution",
        labels={'phase': 'Journey Phase', 'count': 'Number of Clusters'},
        color_discrete_map={
            "Early (Research/Awareness)": "#4CAF50",
            "Middle (Consideration)": "#2196F3",
            "Late (Decision/Purchase)": "#FF9800",
            "Mixed": "#9E9E9E"
        }
    )
    
    # Improve layout
    fig.update_layout(
        xaxis_title="Customer Journey Phase",
        yaxis_title="Number of Clusters",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

@st.cache_data
def create_intent_scores_chart(scores):
    """
    Create a bar chart for intent scores of a specific cluster.
    
    Args:
        scores (dict): Dictionary of intent scores
        
    Returns:
        plotly.graph_objects.Figure: Bar chart of intent scores
    """
    # Create dataframe
    scores_df = pd.DataFrame({
        'intent': list(scores.keys()),
        'score': list(scores.values())
    })
    
    # Create bar chart
    fig = px.bar(
        scores_df,
        x='intent',
        y='score',
        color='intent',
        title="Search Intent Scores",
        labels={'intent': 'Intent Type', 'score': 'Score (%)'},
        color_discrete_map={
            'Informational': '#2196F3',
            'Commercial': '#9C27B0',
            'Transactional': '#FF9800',
            'Navigational': '#4CAF50'
        }
    )
    
    # Set y-axis range to 0-100 for percentage
    fig.update_layout(
        yaxis_range=[0, 100],
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig
