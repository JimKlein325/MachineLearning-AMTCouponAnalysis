"""
Utilities package for coupon analysis project.
Contains data cleaning and investigation functions for machine learning analysis.
"""

# Import key functions for easy access
from .data_cleaning_utils import analyze_missing_data, safe_drop_columns
from .investigation_utils import get_unique_values
from .plot_histogram import plot_categorical_histogram

# What gets imported with "from utils import *"
__all__ = [
    'analyze_missing_data',
    'safe_drop_columns', 
    'get_unique_values',
    'plot_categorical_histogram'
]


