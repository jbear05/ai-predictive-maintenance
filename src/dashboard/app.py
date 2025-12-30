"""
Main Streamlit application entry point.

Run with: streamlit run src/dashboard/app.py
Or from project root: python run_dashboard.py
"""

import logging
import sys
from pathlib import Path

# Add project root and src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

import streamlit as st
from loaders import load_inference_artifacts
from src.ai.inference import predict_failure
from config import config
import warnings

# Import dashboard modules (absolute imports for direct execution)
from dashboard.state import get_state
from dashboard.validation import validate_uploaded_file, validate_uploaded_data
from dashboard.results import display_prediction_results

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@st.cache_resource
def load_model_artifacts():
    """Load model, scaler, and columns once and cache them."""
    logger.info("Loading model artifacts...")
    with st.spinner("Loading model artifacts..."):
        try:
            model, scaler, columns_to_scale, all_features = load_inference_artifacts(config)
            logger.info(f"Model loaded successfully. Features: {len(all_features)}, Columns to scale: {len(columns_to_scale)}")
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}", exc_info=True)
            st.error(f"Error loading model artifacts: {e}")
            st.stop()
    return model, scaler, columns_to_scale, all_features


@st.cache_data
def _group_features(columns: tuple) -> dict:
    """Group feature columns by type. Cached to avoid recomputation."""
    groups = {
        'Rolling Averages': [c for c in columns if 'roll_avg' in c],
        'Rate of Change': [c for c in columns if 'rate_change' in c],
        'EMAs': [c for c in columns if 'ema' in c],
        'Rolling Std Dev': [c for c in columns if 'roll_std' in c],
        'Baseline Deviation': [c for c in columns if 'dev_baseline' in c],
        'Aggregates': [c for c in columns if c.startswith('sensor_') and c.split('_')[1] in ['mean', 'std', 'max', 'min']],
        'Other': []
    }
    all_grouped = sum(groups.values(), [])
    groups['Other'] = [c for c in columns if c not in all_grouped]
    return groups


def _render_sidebar() -> float:
    """Render sidebar and return threshold value."""
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
    
    return threshold


def _render_upload_section(columns_to_scale: list) -> tuple:
    """
    Render file upload and validation section.
    
    Returns (df, is_valid) or (None, False) if no file uploaded.
    """
    st.markdown("### 📁 Upload Sensor Data")
    
    uploaded_file = st.file_uploader(
        "Select your engineered sensor data CSV file",
        type=['csv'],
        help="Upload a CSV with engineered features (rolling averages, EMAs, etc.). Use processed data from the feature engineering pipeline."
    )
    
    if uploaded_file is None:
        return None, False
    
    # Load the data
    logger.info(f"File uploaded: {uploaded_file.name}, size: {uploaded_file.size} bytes")
    try:
        df = validate_uploaded_file(uploaded_file)
        logger.info(f"File validated successfully. Shape: {df.shape}")
    except ValueError as e:
        logger.warning(f"File validation failed: {e}")
        st.error(f"❌ {e}")
        return None, False
    
    # Data preview section
    with st.expander("📋 Data Preview", expanded=False):
        st.dataframe(df.head(10), width="stretch")
        
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
        logger.warning(f"Column validation failed. Missing {len(missing_cols)} columns: {missing_cols[:5]}...")
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
    
    return df, True


def _render_instructions(columns_to_scale: list) -> None:
    """Render getting started instructions when no file is uploaded."""
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
    
    with st.expander("🔍 View All Required Feature Columns"):
        st.write(f"The model requires **{len(columns_to_scale)}** feature columns:")
        
        feature_groups = _group_features(tuple(columns_to_scale))
        
        for group_name, features in feature_groups.items():
            if features:
                st.markdown(f"**{group_name}** ({len(features)} features)")
                with st.expander(f"View {group_name} features"):
                    for feat in features[:20]:
                        st.text(f"- {feat}")
                    if len(features) > 20:
                        st.caption(f"... and {len(features) - 20} more")


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Predictive Maintenance Dashboard",
        page_icon="🔧",
        layout="wide"
    )
    
    # Sidebar
    threshold = _render_sidebar()
    
    # Main header
    st.title("🔧 AI-Powered Predictive Maintenance System")
    st.markdown("### Equipment Health Monitoring & Failure Prediction")
    st.info("ℹ️ This dashboard uses machine learning to predict equipment failures 48 cycles in advance based on sensor data.")
    
    logger.info("Dashboard initialized")
    
    # Load model artifacts (cached)
    try:
        model, scaler, columns_to_scale, all_features = load_model_artifacts()
        st.success("✅ Model loaded successfully")
    except Exception as e:
        logger.critical(f"Failed to load model on startup: {e}", exc_info=True)
        st.error(f"❌ Failed to load model: {e}")
        st.exception(e)
        st.stop()
    
    st.divider()
    
    # File upload section
    df, has_file = _render_upload_section(columns_to_scale)
    
    if not has_file:
        _render_instructions(columns_to_scale)
        return
    
    # Run prediction button
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        run_prediction = st.button("Predict Equipment Health", type="primary", width="stretch")
    
    if run_prediction:
        logger.info(f"Starting prediction on {len(df)} samples with threshold {threshold}")
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
                
                # Store results in typed state container
                state = get_state()
                state.store_results(predictions, probabilities, df, threshold)
                
                num_at_risk = int(predictions.sum())
                logger.info(f"Prediction complete. At risk: {num_at_risk}/{len(predictions)} ({num_at_risk/len(predictions)*100:.1f}%)")
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                logger.error(f"Prediction failed: {e}", exc_info=True)
                st.error(f"❌ Prediction failed: {e}")
                st.exception(e)
    
    # Display results
    state = get_state()
    if state.has_results:
        # Recalculate predictions if threshold changed (allows live adjustment)
        if state.threshold != threshold:
            state.predictions = (state.probabilities >= threshold).astype(int)
            state.threshold = threshold
        display_prediction_results(state, model, all_features, threshold)


if __name__ == "__main__":
    main()
