"""
Data validation utilities for uploaded files.
"""

import pandas as pd
from typing import Set, Tuple, List

# Columns that are expected but not scaled (metadata/identifiers)
METADATA_COLUMNS: Set[str] = {
    'target', 'RUL', 'unit_id', 'time_cycles', 'source_file', 
    'setting_1', 'setting_2', 'setting_3', 'cycle_normalized'
}

# File size limits
MAX_FILE_SIZE_MB = 100
MAX_MEMORY_MB = 500
MAX_COLUMNS = 500
MAX_ROWS = 1_000_000


def validate_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Safely load and validate uploaded CSV.
    
    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object
        
    Returns
    -------
    pd.DataFrame
        Validated dataframe
        
    Raises
    ------
    ValueError
        If file fails validation
    """
    max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    
    if uploaded_file.size > max_size_bytes:
        raise ValueError(f"File too large. Max size: {MAX_FILE_SIZE_MB}MB")
    
    try:
        df = pd.read_csv(
            uploaded_file,
            nrows=MAX_ROWS,
            low_memory=False,
            encoding='utf-8'
        )
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {str(e)}")
    
    if len(df.columns) > MAX_COLUMNS:
        raise ValueError(f"Too many columns (max {MAX_COLUMNS})")
    
    memory_bytes = df.memory_usage(deep=True).sum()
    if memory_bytes > MAX_MEMORY_MB * 1024 * 1024:
        raise ValueError(f"Dataset too large for processing (max {MAX_MEMORY_MB}MB)")
    
    return df


def validate_uploaded_data(
    df: pd.DataFrame, 
    columns_to_scale: List[str]
) -> Tuple[bool, List[str], List[str], List[str]]:
    """
    Validate that uploaded CSV has required engineered feature columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Uploaded dataframe
    columns_to_scale : List[str]
        Required feature columns
        
    Returns
    -------
    Tuple containing:
        - is_valid: Whether all required columns are present
        - missing_columns: List of missing required columns
        - metadata_columns: List of recognized metadata columns
        - unknown_columns: List of unrecognized extra columns
    """
    required_features = set(columns_to_scale) | {'cycle_normalized'}
    uploaded = set(df.columns)
    
    missing = required_features - uploaded
    extra = uploaded - required_features
    metadata_present = extra & METADATA_COLUMNS
    unknown = extra - METADATA_COLUMNS
    
    is_valid = len(missing) == 0
    return is_valid, list(missing), list(metadata_present), list(unknown)
