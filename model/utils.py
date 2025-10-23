import pandas as pd
import numpy as np
from difflib import get_close_matches

def convert_to_scale_6(score_10):
    """Chuyển đổi điểm hệ 10 sang hệ 6 theo công thức: hệ 6 = hệ 10 * 0.6"""
    if pd.isna(score_10):
        return np.nan
    return score_10 * 0.6

def convert_to_numeric(value):
    """Convert value to numeric score on scale 10, return 0 if conversion not possible."""
    try:
        if pd.isna(value):
            return 0
        if isinstance(value, str):
            if value.upper() == 'VT':  # Absent
                return 0
            value = value.replace(',', '.')
        return float(value)
    except (ValueError, TypeError):
        return 0

def suggest_similar(input_value, valid_list, num_suggestions=3):
    """Suggest similar values from a list"""
    matches = get_close_matches(str(input_value), map(str, valid_list), n=num_suggestions, cutoff=0.6)
    return matches

def safe_float(x):
    """Safely convert value to float"""
    try:
        if pd.isna(x):
            return 0.0
        if isinstance(x, str):
            if x.upper() == 'VT':
                return 0.0
            x = x.replace(',', '.')
        return float(x)
    except (ValueError, TypeError):
        return 0.0 