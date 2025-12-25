import streamlit as st
from loaders import load_inference_artifacts
from inference import predict_failure
from config import config
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')


@st.cache_resource
def load_model_artifacts():
    """Load model, scaler, and columns once and cache them."""
    with st.spinner("Loading model artifacts..."):
        try:
            model, scaler, columns_to_scale, all_features = load_inference_artifacts(config)
        except Exception as e:
            st.error(f"Error loading model artifacts: {e}")
            st.stop()
    return model, scaler, columns_to_scale, all_features


# Columns that are expected but not scaled (metadata/identifiers)
METADATA_COLUMNS = {'target', 'RUL', 'unit_id', 'time_cycles', 'source_file', 
                    'setting_1', 'setting_2', 'setting_3', 'cycle_normalized'}


# ✅ BETTER: Validate uploaded files
def validate_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Safely load and validate uploaded CSV."""
    
    # 1. Check file size (prevent DoS)
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    if uploaded_file.size > MAX_FILE_SIZE:
        raise ValueError(f"File too large. Max size: {MAX_FILE_SIZE/1024/1024}MB")
    
    # 2. Load with safety limits
    try:
        df = pd.read_csv(
            uploaded_file,
            nrows=1_000_000,  # Limit rows
            low_memory=False,
            encoding='utf-8'
        )
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {str(e)}")
    
    # 3. Validate structure
    if len(df.columns) > 500:
        raise ValueError("Too many columns")
    
    if df.memory_usage(deep=True).sum() > 500 * 1024 * 1024:  # 500MB
        raise ValueError("Dataset too large for processing")
    
    return df

def validate_uploaded_data(df: pd.DataFrame, columns_to_scale: list) -> tuple[bool, list, list, list]:
    """
    Validate that uploaded CSV has required engineered feature columns.
    
    Returns:
        (is_valid, missing_columns, metadata_columns, unknown_columns)
    """
    required_features = set(columns_to_scale) | {'cycle_normalized'}
    uploaded = set(df.columns)
    
    missing = required_features - uploaded
    extra = uploaded - required_features
    metadata_present = extra & METADATA_COLUMNS
    unknown = extra - METADATA_COLUMNS
    
    is_valid = len(missing) == 0
    return is_valid, list(missing), list(metadata_present), list(unknown)


def create_risk_gauge(probability: float, title: str = "Risk Level") -> go.Figure:
    """Create a gauge chart for risk level."""
    if probability < 0.3:
        color = "#4CAF50"
        risk_level = "Low"
    elif probability < 0.7:
        color = "#FFC107"
        risk_level = "Medium"
    else:
        color = "#FF4B4B"
        risk_level = "High"
    
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


def create_feature_importance_chart(model, columns_to_scale, top_n=10):
    """Create horizontal bar chart of top feature importances."""
    importance_df = pd.DataFrame({
        'Feature': columns_to_scale,
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


def create_sensor_timeseries(df: pd.DataFrame, unit_ids: list, sensor_cols: list):
    """Create time series plots for selected units and sensors."""
    if 'time_cycles' not in df.columns or 'unit_id' not in df.columns:
        return None
    
    # Filter to selected units
    plot_df = df[df['unit_id'].isin(unit_ids)].copy()
    
    if len(plot_df) == 0:
        return None
    
    # Create subplots for first 3 sensors
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[col.replace('_', ' ').title() for col in sensor_cols[:3]]
    )
    
    colors = px.colors.qualitative.Set2
    
    for i, sensor in enumerate(sensor_cols[:3], 1):
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


def create_traffic_light(risk_level: str):
    """Create a traffic light status indicator."""
    if risk_level == "Low":
        return "🟢", "System Normal", "success"
    elif risk_level == "Medium":
        return "🟡", "Attention Required", "warning"
    else:
        return "🔴", "Critical Alert", "error"


def main():
    st.set_page_config(
        page_title="Predictive Maintenance Dashboard",
        page_icon="🔧",
        layout="wide"
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        threshold = st.slider(
            "Failure Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Predictions above this threshold are classified as 'At Risk'"
        )
        
        st.divider()
        
        st.header("📖 About")
        st.write("""
        This dashboard predicts equipment failures up to 48 cycles in advance using AI.
        
        **Features:**
        - Real-time health monitoring
        - Predictive failure detection
        - Feature importance analysis
        - Sensor trend visualization
        """)
        
        st.divider()
        
        st.caption("Built with Streamlit & XGBoost")
    
    # Main header
    st.title("🔧 AI-Powered Predictive Maintenance System")
    st.markdown("### Equipment Health Monitoring & Failure Prediction")
    
    # Info banner
    st.info("ℹ️ This dashboard uses machine learning to predict equipment failures 48 cycles in advance based on sensor data.")
    
    # Load model artifacts (cached)
    try:
        model, scaler, columns_to_scale, all_features = load_model_artifacts()
        model_loaded = True
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.exception(e)
        model_loaded = False
        st.stop()
    
    st.divider()
    
    # File upload section
    st.markdown("### 📁 Upload Sensor Data")
    
    uploaded_file = st.file_uploader(
        "Select your engineered sensor data CSV file",
        type=['csv'],
        help="Upload a CSV with engineered features (rolling averages, EMAs, etc.). Use processed data from the feature engineering pipeline."
    )
    
    if uploaded_file is not None:
        # Load the data
        df = validate_uploaded_file(uploaded_file)
        
        # Data preview section
        with st.expander("📋 Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", f"{len(df.columns)}")
            with col3:
                st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        
        # Validate columns
        is_valid, missing_cols, metadata_cols, unknown_cols = validate_uploaded_data(df, columns_to_scale)
        
        if not is_valid:
            st.error(f"❌ **Validation Error:** {len(missing_cols)} required columns are missing")
            with st.expander("View missing columns"):
                for col in missing_cols:
                    st.write(f"- {col}")
            st.info(f"💡 Your data needs {len(columns_to_scale) + 1} engineered feature columns. Upload data that has been processed through the feature engineering pipeline.")
            st.stop()
        
        st.success("✅ All required feature columns present!")
        
        if metadata_cols:
            st.info(f"📋 Metadata columns detected: {', '.join(metadata_cols[:5])}{'...' if len(metadata_cols) > 5 else ''}")
        
        if unknown_cols:
            with st.expander(f"⚠️ {len(unknown_cols)} unknown columns detected (will be ignored)"):
                st.write(unknown_cols)
        
        # Run prediction
        st.divider()
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            run_prediction = st.button("🔮 Predict Equipment Health", type="primary", use_container_width=True)
        
        if run_prediction:
            with st.spinner("Running inference pipeline..."):
                try:
                    predictions, probabilities = predict_failure(
                        df,
                        model,
                        scaler,
                        columns_to_scale,
                        return_probability=True
                    )
                    
                    # Adjust predictions based on custom threshold
                    predictions = (probabilities >= threshold).astype(int)
                    
                    st.session_state['predictions'] = predictions
                    st.session_state['probabilities'] = probabilities
                    st.session_state['original_df'] = df
                    st.session_state['threshold'] = threshold
                    
                    st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    st.exception(e)
        
        # Display results
        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            probabilities = st.session_state['probabilities']
            original_df = st.session_state['original_df']
            
            st.divider()
            st.header("📊 Prediction Results")
            
            # Calculate metrics
            num_failures = int(predictions.sum())
            num_healthy = int((predictions == 0).sum())
            avg_prob = float(probabilities.mean())
            max_prob = float(probabilities.max())
            
            # Determine overall fleet risk
            if avg_prob < 0.3:
                overall_risk = "Low"
            elif avg_prob < 0.7:
                overall_risk = "Medium"
            else:
                overall_risk = "High"
            
            # Status banner
            if num_failures > 0:
                st.error(f"⚠️ **ALERT:** {num_failures} unit(s) at risk of failure within 48 cycles!")
            else:
                st.success("✅ **ALL SYSTEMS HEALTHY:** No immediate failures predicted")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Prediction Confidence",
                    f"{avg_prob:.1%}",
                    help="Average failure probability across all units"
                )
            
            with col2:
                fleet_health = (1 - avg_prob) * 100
                st.metric(
                    "Fleet Health Score",
                    f"{fleet_health:.1f}/100",
                    help="Overall equipment fleet health (inverse of avg risk)"
                )
            
            with col3:
                st.metric(
                    "Risk Level",
                    overall_risk,
                    help="Overall fleet risk assessment"
                )
            
            with col4:
                pass  # Placeholder for symmetry
            
            st.divider()
            
            # Second metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🔴 At Risk",
                    f"{num_failures}",
                    delta=f"{num_failures/len(predictions)*100:.1f}%",
                    delta_color="inverse",
                    help=f"Units with probability ≥ {threshold:.0%}"
                )
            
            with col2:
                st.metric(
                    "🟢 Healthy",
                    f"{num_healthy}",
                    delta=f"{num_healthy/len(predictions)*100:.1f}%",
                    help=f"Units with probability < {threshold:.0%}"
                )
            
            with col3:
                st.metric(
                    "Avg Risk Score",
                    f"{avg_prob:.1%}",
                    help="Mean failure probability"
                )
            
            with col4:
                st.metric(
                    "Max Risk Score",
                    f"{max_prob:.1%}",
                    help="Highest individual unit risk"
                )
            
            st.divider()
            
            # Visualizations section
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Risk Score Distribution")
                fig_hist = px.histogram(
                    x=probabilities,
                    nbins=30,
                    labels={'x': 'Failure Probability', 'y': 'Count'},
                    color_discrete_sequence=['#1f77b4']
                )
                fig_hist.add_vline(
                    x=threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold ({threshold:.0%})",
                    annotation_position="top right"
                )
                fig_hist.update_layout(
                    xaxis_title="Failure Probability",
                    yaxis_title="Number of Units",
                    showlegend=False,
                    height=350
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                st.subheader("Equipment Health Overview")
                gauge = create_risk_gauge(avg_prob, "Fleet Risk Level")
                st.plotly_chart(gauge, use_container_width=True)
            
            # Feature importance
            st.divider()
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("Top Contributing Factors")
                fig_importance = create_feature_importance_chart(model, all_features, top_n=10)
                st.plotly_chart(fig_importance, use_container_width=True)
            
            with col2:
                st.subheader("Status Indicator")
                st.write("")  # Spacing
                st.write("")  # Spacing
                icon, status_text, status_type = create_traffic_light(overall_risk)
                st.markdown(f"# {icon}")
                if status_type == "success":
                    st.success(status_text)
                elif status_type == "warning":
                    st.warning(status_text)
                else:
                    st.error(status_text)
            
            # Time series visualization
            if 'unit_id' in original_df.columns and 'time_cycles' in original_df.columns:
                st.divider()
                st.subheader("📈 Sensor Readings Over Time (High-Risk Units)")
                
                # Get top 3 high-risk units
                results_df = original_df.copy()
                results_df['Failure_Probability'] = probabilities
                high_risk_df = results_df.sort_values('Failure_Probability', ascending=False)
                high_risk_units = high_risk_df['unit_id'].unique()[:3].tolist()
                
                # Find sensor columns (raw sensors, not engineered features)
                sensor_cols = [col for col in original_df.columns if col.startswith('sensor_') and '_' not in col[7:]]
                if not sensor_cols:
                    # Fallback to any numeric columns
                    sensor_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()[:3]
                
                if len(high_risk_units) > 0 and len(sensor_cols) > 0:
                    fig_timeseries = create_sensor_timeseries(original_df, high_risk_units, sensor_cols)
                    if fig_timeseries:
                        st.plotly_chart(fig_timeseries, use_container_width=True)
                    else:
                        st.info("Unable to generate time series plots - insufficient data")
                else:
                    st.info("Time series visualization unavailable - no suitable sensor columns found")
            
            # Detailed results table
            st.divider()
            st.subheader("Detailed Results")
            
            results_df = original_df.copy()
            results_df['Prediction'] = predictions
            results_df['Failure_Probability'] = probabilities
            results_df['Risk_Level'] = pd.cut(
                probabilities,
                bins=[0, 0.3, 0.7, 1.0],
                labels=['🟢 Low', '🟡 Medium', '🔴 High']
            )
            
            # Sort by risk (highest first)
            results_df = results_df.sort_values('Failure_Probability', ascending=False)
            
            # Filter options
            col1, col2 = st.columns([1, 3])
            with col1:
                risk_filter = st.multiselect(
                    "Filter by Risk Level",
                    options=['🟢 Low', '🟡 Medium', '🔴 High'],
                    default=['🟡 Medium', '🔴 High']
                )
            
            if risk_filter:
                filtered_df = results_df[results_df['Risk_Level'].isin(risk_filter)]
            else:
                filtered_df = results_df
            
            # Show key columns first
            display_cols = ['unit_id', 'time_cycles', 'Failure_Probability', 'Risk_Level', 'Prediction']
            available_display_cols = [c for c in display_cols if c in filtered_df.columns]
            
            if available_display_cols:
                st.dataframe(
                    filtered_df[available_display_cols].head(100),
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
            else:
                st.dataframe(
                    filtered_df[['Failure_Probability', 'Risk_Level', 'Prediction']].head(100),
                    use_container_width=True,
                    height=400
                )
            
            st.caption(f"Showing {len(filtered_df)} of {len(results_df)} total predictions")
            
            # Download section
            st.divider()
            st.subheader("📥 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Full Results as CSV",
                    data=csv,
                    file_name="predictions_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                high_risk_csv = results_df[results_df['Prediction'] == 1].to_csv(index=False)
                st.download_button(
                    label="Download High-Risk Units Only",
                    data=high_risk_csv,
                    file_name="high_risk_units.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    else:
        # No file uploaded - show instructions
        st.divider()
        st.markdown("### 📖 Getting Started")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Expected Data Format
            
            Your CSV file must contain **engineered features**, not raw sensor data.
            
            The model expects features including:
            - Rolling averages (3, 5, 10 cycles)
            - Rate of change indicators
            - Exponential moving averages
            - Rolling standard deviation
            - Deviation from baseline
            - Cross-sensor aggregates
            - Cycle progression indicator
            
            📄 **Tip:** Use processed data from `data/processed/` as a template.
            """)
        
        with col2:
            st.markdown("""
            #### How to Use
            
            1. **Upload** your engineered sensor data CSV
            2. **Review** the data preview and validation
            3. **Click** "Predict Equipment Health"
            4. **Analyze** results and risk scores
            5. **Download** predictions for further action
            
            💡 **Risk Levels:**
            - 🟢 Low (0-30%): Normal operation
            - 🟡 Medium (30-70%): Monitor closely
            - 🔴 High (70-100%): Immediate attention required
            """)
        
        if model_loaded:
            with st.expander("🔍 View All Required Feature Columns"):
                st.write(f"The model requires **{len(columns_to_scale)}** feature columns:")
                
                # Group features by type
                feature_groups = {
                    'Rolling Averages': [c for c in columns_to_scale if 'roll_avg' in c],
                    'Rate of Change': [c for c in columns_to_scale if 'rate_change' in c],
                    'EMAs': [c for c in columns_to_scale if 'ema' in c],
                    'Rolling Std Dev': [c for c in columns_to_scale if 'roll_std' in c],
                    'Baseline Deviation': [c for c in columns_to_scale if 'dev_baseline' in c],
                    'Aggregates': [c for c in columns_to_scale if c.startswith('sensor_') and c.split('_')[1] in ['mean', 'std', 'max', 'min']],
                    'Other': []
                }
                
                # Catch remaining features
                all_grouped = sum(feature_groups.values(), [])
                feature_groups['Other'] = [c for c in columns_to_scale if c not in all_grouped]
                
                for group_name, features in feature_groups.items():
                    if features:
                        st.markdown(f"**{group_name}** ({len(features)} features)")
                        with st.expander(f"View {group_name} features"):
                            for feat in features[:20]:  # Show first 20
                                st.text(f"- {feat}")
                            if len(features) > 20:
                                st.caption(f"... and {len(features) - 20} more")


if __name__ == "__main__":
    main()