"""
Risk level classification logic.

Centralizes the repeated risk level threshold logic used throughout the dashboard.
"""

from enum import Enum
from typing import Tuple
import pandas as pd
import numpy as np


class RiskLevel(Enum):
    """
    Equipment risk level classification.
    
    Each value contains: (label, icon, status_type, low_threshold, high_threshold)
    """
    LOW = ("Low", "🟢", "success", 0.0, 0.3)
    MEDIUM = ("Medium", "🟡", "warning", 0.3, 0.7)
    HIGH = ("High", "🔴", "error", 0.7, 1.0)
    
    @property
    def label(self) -> str:
        return self.value[0]
    
    @property
    def icon(self) -> str:
        return self.value[1]
    
    @property
    def status_type(self) -> str:
        """Returns 'success', 'warning', or 'error' for Streamlit status."""
        return self.value[2]
    
    @property
    def display_label(self) -> str:
        """Returns icon + label for display (e.g., '🟢 Low')."""
        return f"{self.icon} {self.label}"
    
    @property
    def status_message(self) -> str:
        """Returns status message for traffic light indicator."""
        messages = {
            "LOW": "System Normal",
            "MEDIUM": "Attention Required", 
            "HIGH": "Critical Alert"
        }
        return messages[self.name]
    
    @classmethod
    def from_probability(cls, probability: float) -> "RiskLevel":
        """
        Classify a probability into a risk level.
        
        Parameters
        ----------
        probability : float
            Failure probability between 0 and 1
            
        Returns
        -------
        RiskLevel
            The corresponding risk level
        """
        if probability < 0.3:
            return cls.LOW
        elif probability < 0.7:
            return cls.MEDIUM
        else:
            return cls.HIGH
    
    @classmethod
    def categorize_series(cls, probabilities: np.ndarray) -> pd.Categorical:
        """
        Categorize an array of probabilities into risk levels.
        
        Parameters
        ----------
        probabilities : np.ndarray
            Array of failure probabilities
            
        Returns
        -------
        pd.Categorical
            Categorical series with risk level labels
        """
        return pd.cut(
            probabilities,
            bins=[0, 0.3, 0.7, 1.0],
            labels=[cls.LOW.display_label, cls.MEDIUM.display_label, cls.HIGH.display_label]
        )
    
    @classmethod
    def get_color(cls, probability: float) -> str:
        """Get the color code for a probability value."""
        level = cls.from_probability(probability)
        colors = {
            cls.LOW: "#4CAF50",
            cls.MEDIUM: "#FFC107",
            cls.HIGH: "#FF4B4B"
        }
        return colors[level]
