"""
Results display component for the dashboard.

Handles all the visualization and display of prediction results.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List

from .state import DashboardState
from .risk import RiskLevel
from .charts import (
    create_risk_gauge,
    create_feature_importance_chart,
    create_sensor_timeseries,
    create_risk_histogram,
    create_mini_gauge,
    create_count_gauge,
    SENSOR_DESCRIPTIONS,
)


def display_prediction_results(
    state: DashboardState,
    model,
    all_features: List[str],
    threshold: float
) -> None:
    """
    Display all prediction results including metrics, charts, and tables.
    
    Parameters
    ----------
    state : DashboardState
        Current dashboard state with predictions
    model : XGBoost model
        Trained model for feature importance
    all_features : List[str]
        Feature names for importance chart
    threshold : float
        Classification threshold
    """
    predictions = state.predictions
    probabilities = state.probabilities
    original_df = state.original_df
    
    st.divider()
    st.header("📊 Prediction Results")
    
    # Calculate metrics
    num_failures = int(predictions.sum())
    num_healthy = int((predictions == 0).sum())
    avg_prob = float(probabilities.mean())
    max_prob = float(probabilities.max())
    overall_risk = RiskLevel.from_probability(avg_prob, threshold)
    
    # Model confidence: how certain the model is (distance from 0.5)
    # 0.5 = completely uncertain, 0 or 1 = fully confident
    model_confidence = float(np.mean(np.abs(probabilities - 0.5) * 2))
    
    # Status banner
    if num_failures > 0:
        st.error(f"⚠️ **ALERT:** {num_failures} unit(s) at risk of failure within 48 cycles!")
    else:
        st.success("✅ **ALL SYSTEMS HEALTHY:** No immediate failures predicted")
    
    # Primary metrics row
    _display_primary_metrics(model_confidence, overall_risk, avg_prob)
    
    # Secondary metrics row
    _display_secondary_metrics(num_failures, num_healthy, avg_prob, max_prob, len(predictions), threshold)
    
    # Charts section
    _display_charts(probabilities, avg_prob, threshold, model, all_features, overall_risk)
    
    # Time series (if applicable)
    _display_time_series(original_df, probabilities)
    
    # Results table
    results_df = _build_results_dataframe(original_df, predictions, probabilities, threshold)
    _display_results_table(results_df)
    
    # Downloads
    _display_downloads(results_df)


def _display_primary_metrics(model_confidence: float, overall_risk: RiskLevel, avg_prob: float) -> None:
    """Display the first row of metrics as gauges."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = create_mini_gauge(
            value=model_confidence * 100,
            title="Model Confidence",
            max_value=100,
            suffix="%",
            invert_color=True  # High confidence is good
        )
        st.plotly_chart(fig, use_container_width=True, key="gauge_confidence")
        st.markdown("<p style='text-align: center; color: gray; font-size: 0.85em;'>ℹ️ How certain the model is in its predictions (higher = more decisive)</p>", unsafe_allow_html=True)
    
    with col2:
        fleet_health = (1 - avg_prob) * 100
        fig = create_mini_gauge(
            value=fleet_health,
            title="Fleet Health Score",
            max_value=100,
            suffix="/100",
            invert_color=True  # High health is good
        )
        st.plotly_chart(fig, use_container_width=True, key="gauge_health")
        st.markdown("<p style='text-align: center; color: gray; font-size: 0.85em;'>ℹ️ Overall equipment fleet health (inverse of avg risk)</p>", unsafe_allow_html=True)
    
    with col3:
        # Risk level gauge (use avg_prob for the gauge value)
        fig = create_mini_gauge(
            value=avg_prob * 100,
            title=f"Risk Level: {overall_risk.label}",
            max_value=100,
            suffix="%",
            invert_color=False  # High risk is bad
        )
        st.plotly_chart(fig, use_container_width=True, key="gauge_risk")
        st.markdown("<p style='text-align: center; color: gray; font-size: 0.85em;'>ℹ️ Overall fleet risk assessment based on threshold</p>", unsafe_allow_html=True)
    
    st.divider()


def _display_secondary_metrics(
    num_failures: int,
    num_healthy: int,
    avg_prob: float,
    max_prob: float,
    total: int,
    threshold: float
) -> None:
    """Display the second row of metrics as gauges."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fig = create_count_gauge(
            value=num_failures,
            total=total,
            title="🔴 At Risk",
            is_risk=True
        )
        st.plotly_chart(fig, use_container_width=True, key="gauge_at_risk")
        st.markdown(f"<p style='text-align: center; color: gray; font-size: 0.85em;'>ℹ️ Units with probability ≥ {threshold:.0%}</p>", unsafe_allow_html=True)
    
    with col2:
        fig = create_count_gauge(
            value=num_healthy,
            total=total,
            title="🟢 Healthy",
            is_risk=False
        )
        st.plotly_chart(fig, use_container_width=True, key="gauge_healthy")
        st.markdown(f"<p style='text-align: center; color: gray; font-size: 0.85em;'>ℹ️ Units with probability < {threshold:.0%}</p>", unsafe_allow_html=True)
    
    with col3:
        fig = create_mini_gauge(
            value=avg_prob * 100,
            title="Avg Risk Score",
            max_value=100,
            suffix="%",
            invert_color=False
        )
        st.plotly_chart(fig, use_container_width=True, key="gauge_avg_risk")
        st.markdown("<p style='text-align: center; color: gray; font-size: 0.85em;'>ℹ️ Mean failure probability across all units</p>", unsafe_allow_html=True)
    
    with col4:
        fig = create_mini_gauge(
            value=max_prob * 100,
            title="Max Risk Score",
            max_value=100,
            suffix="%",
            invert_color=False
        )
        st.plotly_chart(fig, use_container_width=True, key="gauge_max_risk")
        st.markdown("<p style='text-align: center; color: gray; font-size: 0.85em;'>ℹ️ Highest individual unit risk</p>", unsafe_allow_html=True)
    
    st.divider()


def _display_charts(
    probabilities: np.ndarray,
    avg_prob: float,
    threshold: float,
    model,
    all_features: List[str],
    overall_risk: RiskLevel
) -> None:
    """Display histogram, feature importance, and status indicator."""
    # Row 1: Histogram (full width now)
    st.subheader("Risk Score Distribution")
    fig_hist = create_risk_histogram(probabilities, threshold)
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.divider()
    
    # Row 2: Feature importance and Status
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Top Contributing Factors")
        fig_importance = create_feature_importance_chart(model, all_features, top_n=10)
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        st.subheader("Status Indicator")
        st.write("")
        st.write("")
        st.markdown(f"# {overall_risk.icon}")
        status_func = getattr(st, overall_risk.status_type)
        status_func(overall_risk.status_message)


def _display_time_series(original_df: pd.DataFrame, probabilities: np.ndarray) -> None:
    """Display time series for high-risk units if data is available."""
    if 'unit_id' not in original_df.columns or 'time_cycles' not in original_df.columns:
        return
    
    st.divider()
    st.subheader("📈 Sensor Readings Over Time (High-Risk Units)")
    
    # Help expander for sensor descriptions
    with st.expander("ℹ️ What do these sensors measure?"):
        st.markdown("**NASA C-MAPSS Turbofan Engine Sensors:**")
        cols = st.columns(2)
        sensors = list(SENSOR_DESCRIPTIONS.items())
        mid = len(sensors) // 2
        with cols[0]:
            for sensor, desc in sensors[:mid]:
                st.markdown(f"- **{sensor}**: {desc}")
        with cols[1]:
            for sensor, desc in sensors[mid:]:
                st.markdown(f"- **{sensor}**: {desc}")
    
    # Get top 3 high-risk units
    temp_df = original_df[['unit_id']].copy()
    temp_df['prob'] = probabilities
    high_risk_units = (
        temp_df.sort_values('prob', ascending=False)['unit_id']
        .unique()[:3]
        .tolist()
    )
    
    # Find sensor columns
    sensor_cols = [
        col for col in original_df.columns 
        if col.startswith('sensor_') and '_' not in col[7:]
    ]
    if not sensor_cols:
        sensor_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()[:3]
    
    if high_risk_units and sensor_cols:
        fig = create_sensor_timeseries(original_df, high_risk_units, sensor_cols)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Unable to generate time series plots - insufficient data")
    else:
        st.info("Time series visualization unavailable - no suitable sensor columns found")


def _build_results_dataframe(
    original_df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5
) -> pd.DataFrame:
    """Build the results dataframe with predictions and risk levels."""
    # Only copy columns we need for display
    keep_cols = ['unit_id', 'time_cycles']
    available_cols = [c for c in keep_cols if c in original_df.columns]
    
    if available_cols:
        results_df = original_df[available_cols].copy()
    else:
        results_df = pd.DataFrame(index=original_df.index)
    
    results_df['Prediction'] = predictions
    results_df['Failure_Probability'] = probabilities
    results_df['Risk_Level'] = RiskLevel.categorize_series(probabilities, threshold)
    
    return results_df.sort_values('Failure_Probability', ascending=False)


def _display_results_table(results_df: pd.DataFrame) -> None:
    """Display the filtered results table."""
    st.divider()
    st.subheader("Detailed Results")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=['🟢 Low', '🟡 Medium', '🔴 High'],
            default=['🟡 Medium', '🔴 High']
        )
    
    filtered_df = results_df[results_df['Risk_Level'].isin(risk_filter)] if risk_filter else results_df
    
    st.dataframe(
        filtered_df.head(100),
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    st.caption(f"Showing {min(100, len(filtered_df))} of {len(results_df)} total predictions")


def _display_downloads(results_df: pd.DataFrame) -> None:
    """Display download buttons for results."""
    st.divider()
    st.subheader("📥 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="Download Full Results as CSV",
            data=results_df.to_csv(index=False),
            file_name="predictions_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        high_risk_df = results_df[results_df['Prediction'] == 1]
        st.download_button(
            label="Download High-Risk Units Only",
            data=high_risk_df.to_csv(index=False),
            file_name="high_risk_units.csv",
            mime="text/csv",
            use_container_width=True
        )
