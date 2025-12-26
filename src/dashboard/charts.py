"""
Chart creation functions for the dashboard.

Pure functions with no Streamlit dependencies (except for display).
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional

from .risk import RiskLevel


def create_risk_gauge(probability: float, title: str = "Risk Level") -> go.Figure:
    """
    Create a gauge chart for risk level visualization.
    
    Parameters
    ----------
    probability : float
        Failure probability between 0 and 1
    title : str
        Chart title
        
    Returns
    -------
    go.Figure
        Plotly gauge chart
    """
    color = RiskLevel.get_color(probability)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18}},
        number={'suffix': "%", 'font': {'size': 32}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def create_feature_importance_chart(
    model,
    feature_names: List[str],
    top_n: int = 10
) -> go.Figure:
    """
    Create horizontal bar chart of top feature importances.
    
    Parameters
    ----------
    model : XGBoost model
        Trained model with feature_importances_ attribute
    feature_names : List[str]
        List of feature names matching model features
    top_n : int
        Number of top features to display
        
    Returns
    -------
    go.Figure
        Plotly horizontal bar chart
    """
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(top_n)
    
    fig = go.Figure(go.Bar(
        x=importance_df['Importance'] * 100,
        y=importance_df['Feature'],
        orientation='h',
        marker=dict(color='#1f77b4')
    ))
    
    fig.update_layout(
        title="Top Contributing Features",
        xaxis_title="Importance (%)",
        yaxis_title="",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_sensor_timeseries(
    df: pd.DataFrame,
    unit_ids: List[int],
    sensor_cols: List[str]
) -> Optional[go.Figure]:
    """
    Create time series plots for selected units and sensors.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing sensor data with 'time_cycles' and 'unit_id'
    unit_ids : List[int]
        List of unit IDs to plot
    sensor_cols : List[str]
        List of sensor column names to plot
        
    Returns
    -------
    Optional[go.Figure]
        Plotly figure with subplots, or None if insufficient data
    """
    if 'time_cycles' not in df.columns or 'unit_id' not in df.columns:
        return None
    
    plot_df = df[df['unit_id'].isin(unit_ids)].copy()
    
    if len(plot_df) == 0:
        return None
    
    # Create subplots for first 3 sensors
    sensors_to_plot = sensor_cols[:3]
    fig = make_subplots(
        rows=1, cols=len(sensors_to_plot),
        subplot_titles=[col.replace('_', ' ').title() for col in sensors_to_plot]
    )
    
    colors = px.colors.qualitative.Set2
    
    for i, sensor in enumerate(sensors_to_plot, 1):
        if sensor not in plot_df.columns:
            continue
            
        for j, unit_id in enumerate(unit_ids):
            unit_data = plot_df[plot_df['unit_id'] == unit_id].sort_values('time_cycles')
            
            fig.add_trace(
                go.Scatter(
                    x=unit_data['time_cycles'],
                    y=unit_data[sensor],
                    mode='lines',
                    name=f'Unit {unit_id}',
                    line=dict(color=colors[j % len(colors)], width=2),
                    legendgroup=f'unit_{unit_id}',
                    showlegend=(i == 1)
                ),
                row=1, col=i
            )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Time Cycles")
    
    return fig


def create_risk_histogram(
    probabilities: np.ndarray,
    threshold: float
) -> go.Figure:
    """
    Create histogram of risk score distribution.
    
    Parameters
    ----------
    probabilities : np.ndarray
        Array of failure probabilities
    threshold : float
        Threshold value to mark on chart
        
    Returns
    -------
    go.Figure
        Plotly histogram
    """
    fig = px.histogram(
        x=probabilities,
        nbins=30,
        labels={'x': 'Failure Probability', 'y': 'Count'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({threshold:.0%})",
        annotation_position="top right"
    )
    fig.update_layout(
        xaxis_title="Failure Probability",
        yaxis_title="Number of Units",
        showlegend=False,
        height=350
    )
    return fig
