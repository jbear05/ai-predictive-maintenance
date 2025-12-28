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


# Sensor descriptions based on NASA C-MAPSS turbofan engine data
SENSOR_DESCRIPTIONS = {
    'sensor_1': 'Total temperature at fan inlet (T1)',
    'sensor_2': 'Total temperature at LPC outlet (T2)',
    'sensor_3': 'Total temperature at HPC outlet (T24)',
    'sensor_4': 'Total temperature at LPT outlet (T30)',
    'sensor_5': 'Total pressure at fan inlet (P1)',
    'sensor_6': 'Total pressure at bypass-duct (P15)',
    'sensor_7': 'Total pressure at HPC outlet (P30)',
    'sensor_8': 'Physical fan speed (Nf)',
    'sensor_9': 'Physical core speed (Nc)',
    'sensor_10': 'Engine pressure ratio (EPR)',
    'sensor_11': 'Static pressure at HPC outlet (Ps30)',
    'sensor_12': 'Ratio of fuel flow to Ps30 (phi)',
    'sensor_13': 'Corrected fan speed (NRf)',
    'sensor_14': 'Corrected core speed (NRc)',
    'sensor_15': 'Bypass ratio (BPR)',
    'sensor_16': 'Burner fuel-air ratio (farB)',
    'sensor_17': 'Bleed enthalpy (htBleed)',
    'sensor_18': 'Demanded fan speed (Nf_dmd)',
    'sensor_19': 'Demanded corrected fan speed (PCNfR_dmd)',
    'sensor_20': 'HPT coolant bleed (W31)',
    'sensor_21': 'LPT coolant bleed (W32)',
}


def get_sensor_description(sensor_name: str) -> str:
    """Get human-readable description for a sensor."""
    # Extract base sensor name (e.g., 'sensor_2' from 'sensor_2_roll_avg_3')
    parts = sensor_name.split('_')
    if len(parts) >= 2 and parts[0] == 'sensor':
        base_name = f"{parts[0]}_{parts[1]}"
        return SENSOR_DESCRIPTIONS.get(base_name, sensor_name)
    return sensor_name


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


def create_mini_gauge(
    value: float,
    title: str,
    max_value: float = 100,
    suffix: str = "%",
    color: str = None,
    invert_color: bool = False
) -> go.Figure:
    """
    Create a compact gauge chart for metric visualization.
    
    Parameters
    ----------
    value : float
        The value to display (0-max_value range)
    title : str
        Chart title
    max_value : float
        Maximum value for the gauge scale
    suffix : str
        Suffix for the number display (e.g., "%", "/100")
    color : str
        Override color for the bar (auto-calculated if None)
    invert_color : bool
        If True, green is high and red is low (for health scores)
        
    Returns
    -------
    go.Figure
        Plotly gauge chart
    """
    # Normalize to 0-1 for color calculation
    normalized = value / max_value if max_value > 0 else 0
    
    if color is None:
        if invert_color:
            # Green for high values (health score)
            if normalized >= 0.7:
                color = "#4CAF50"  # Green
            elif normalized >= 0.3:
                color = "#FFC107"  # Yellow
            else:
                color = "#F44336"  # Red
        else:
            # Red for high values (risk score)
            if normalized < 0.3:
                color = "#4CAF50"  # Green
            elif normalized < 0.7:
                color = "#FFC107"  # Yellow
            else:
                color = "#F44336"  # Red
    
    # Define steps based on invert_color
    if invert_color:
        steps = [
            {'range': [0, max_value * 0.3], 'color': "lightcoral"},
            {'range': [max_value * 0.3, max_value * 0.7], 'color': "lightyellow"},
            {'range': [max_value * 0.7, max_value], 'color': "lightgreen"}
        ]
    else:
        steps = [
            {'range': [0, max_value * 0.3], 'color': "lightgreen"},
            {'range': [max_value * 0.3, max_value * 0.7], 'color': "lightyellow"},
            {'range': [max_value * 0.7, max_value], 'color': "lightcoral"}
        ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14}},
        number={'suffix': suffix, 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, max_value], 'tickfont': {'size': 10}},
            'bar': {'color': color, 'thickness': 0.7},
            'steps': steps,
            'borderwidth': 1,
            'bordercolor': "gray"
        }
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=15, r=15, t=40, b=15)
    )
    return fig


def create_count_gauge(
    value: int,
    total: int,
    title: str,
    is_risk: bool = True
) -> go.Figure:
    """
    Create a gauge for count metrics (At Risk / Healthy).
    
    Parameters
    ----------
    value : int
        Count value
    total : int
        Total count for percentage
    title : str
        Chart title
    is_risk : bool
        If True, high values are bad (red). If False, high values are good (green).
        
    Returns
    -------
    go.Figure
        Plotly gauge chart
    """
    percentage = (value / total * 100) if total > 0 else 0
    
    if is_risk:
        color = "#F44336" if percentage > 30 else ("#FFC107" if percentage > 10 else "#4CAF50")
    else:
        color = "#4CAF50" if percentage > 70 else ("#FFC107" if percentage > 30 else "#F44336")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={'reference': 0, 'relative': False, 'valueformat': '.1f', 'suffix': f' ({percentage:.1f}%)'},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14}},
        number={'font': {'size': 28}},
        gauge={
            'axis': {'range': [0, total], 'tickfont': {'size': 10}},
            'bar': {'color': color, 'thickness': 0.7},
            'steps': [
                {'range': [0, total], 'color': "lightgray"}
            ],
            'borderwidth': 1,
            'bordercolor': "gray"
        }
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=15, r=15, t=40, b=15)
    )
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
    
    # Get sensor descriptions for subplot titles
    subplot_titles = []
    for col in sensors_to_plot:
        desc = get_sensor_description(col)
        # Shorten description for title if too long
        if len(desc) > 30:
            desc = desc[:27] + "..."
        subplot_titles.append(desc)
    
    fig = make_subplots(
        rows=1, cols=len(sensors_to_plot),
        subplot_titles=subplot_titles
    )
    
    colors = px.colors.qualitative.Set2
    
    for i, sensor in enumerate(sensors_to_plot, 1):
        if sensor not in plot_df.columns:
            continue
        
        sensor_desc = get_sensor_description(sensor)
            
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
                    showlegend=(i == 1),
                    hovertemplate=f"<b>{sensor_desc}</b><br>Unit {unit_id}<br>Cycle: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>"
                ),
                row=1, col=i
            )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='closest'
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
