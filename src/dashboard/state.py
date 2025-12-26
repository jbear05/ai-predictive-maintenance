"""
Type-safe session state management for the dashboard.

Replaces scattered st.session_state dict access with a typed container.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st


@dataclass
class DashboardState:
    """
    Centralized state container for dashboard data.
    
    Benefits:
    - IDE autocomplete and type checking
    - Single source of truth for state shape
    - Easy to test and mock
    """
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    original_df: Optional[pd.DataFrame] = None
    threshold: float = 0.5
    
    @property
    def has_results(self) -> bool:
        """Check if prediction results are available."""
        return self.predictions is not None and self.probabilities is not None
    
    def store_results(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        original_df: pd.DataFrame,
        threshold: float
    ) -> None:
        """Store prediction results in state."""
        self.predictions = predictions
        self.probabilities = probabilities
        self.original_df = original_df
        self.threshold = threshold
    
    def clear(self) -> None:
        """Clear all stored results."""
        self.predictions = None
        self.probabilities = None
        self.original_df = None


def get_state() -> DashboardState:
    """
    Get or initialize the dashboard state from Streamlit session.
    
    Returns
    -------
    DashboardState
        The current dashboard state instance
    """
    if 'dashboard_state' not in st.session_state:
        st.session_state.dashboard_state = DashboardState()
    return st.session_state.dashboard_state
