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
    def from_probability(cls, probability: float, threshold: float = 0.5) -> "RiskLevel":
        """
        Classify a probability into a risk level.
        
        Risk levels are dynamically calculated based on the failure threshold:
        - Low: 0 to threshold * 0.6
        - Medium: threshold * 0.6 to threshold
        - High: threshold and above
        
        Parameters
        ----------
        probability : float
            Failure probability between 0 and 1
        threshold : float
            The failure threshold (probabilities >= threshold are "At Risk")
            
        Returns
        -------
        RiskLevel
            The corresponding risk level
        """
        # Handle edge cases
        if threshold <= 0.01:
            return cls.HIGH
        elif threshold >= 0.99:
            if probability < 0.5:
                return cls.LOW
            else:
                return cls.MEDIUM
        
        # Dynamic thresholds based on slider
        low_upper = threshold * 0.6
        
        if probability < low_upper:
            return cls.LOW
        elif probability < threshold:
            return cls.MEDIUM
        else:
            return cls.HIGH
    
    @classmethod
    def categorize_series(cls, probabilities: np.ndarray, threshold: float = 0.5) -> pd.Categorical:
        """
        Categorize an array of probabilities into risk levels.
        
        Risk levels are dynamically calculated based on the failure threshold:
        - Low: 0 to threshold * 0.6
        - Medium: threshold * 0.6 to threshold
        - High: threshold to 1.0
        
        Parameters
        ----------
        probabilities : np.ndarray
            Array of failure probabilities
        threshold : float
            The failure threshold (probabilities >= threshold are "At Risk")
            
        Returns
        -------
        pd.Categorical
            Categorical series with risk level labels
        """
        # Handle edge cases for threshold at 0 or 1
        if threshold <= 0.01:
            # Everything is High risk when threshold is ~0
            return pd.Categorical(
                [cls.HIGH.display_label] * len(probabilities),
                categories=[cls.LOW.display_label, cls.MEDIUM.display_label, cls.HIGH.display_label]
            )
        elif threshold >= 0.99:
            # Nothing is High risk when threshold is ~1, split into Low/Medium
            result = pd.cut(
                probabilities,
                bins=[0, 0.5, 1.0],
                labels=[cls.LOW.display_label, cls.MEDIUM.display_label],
                include_lowest=True
            )
            return result.add_categories([cls.HIGH.display_label])
        
        # Calculate dynamic boundaries based on threshold
        # Low: 0 to 60% of threshold
        # Medium: 60% of threshold to threshold
        # High: threshold and above
        low_upper = threshold * 0.6
        medium_upper = threshold
        
        return pd.cut(
            probabilities,
            bins=[0, low_upper, medium_upper, 1.0],
            labels=[cls.LOW.display_label, cls.MEDIUM.display_label, cls.HIGH.display_label],
            include_lowest=True
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
