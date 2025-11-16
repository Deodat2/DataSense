# ============================================================
# dataset_analyzer/content_analysis.py
# ============================================================
# This module provides helper functions to analyze structured
# dataset contents such as DataFrames (CSV, Excel) or JSON data.
# It detects the learning type (Supervised, Semi-supervised,
# or Unsupervised) and extracts potential class labels.
# ============================================================
import logging
import pandas as pd
from typing import Any, Tuple, List

from .models.labels import (
    detect_labels_dataframe_smart,
    detect_classes_dataframe,
    find_labels_json_recursive
)

# -------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# 1. DataFrame Analysis
# ------------------------------------------------------------
def analyze_dataframe(df: pd.DataFrame) -> Tuple[str, List[Any]]:
    """
    Analyze a pandas DataFrame to infer the learning type and class labels.

    Args:
        df (pd.DataFrame): The dataset to analyze.

    Returns:
        tuple:
            (str, list)
            - str: Detected learning type ("Supervised", "Semi-supervised", or "Unsupervised").
            - list: List of detected class labels (if any).
    """
    try:
        # Detect learning type using label column patterns
        learning_type = detect_labels_dataframe_smart(df)

        # Extract possible class names from the candidate label column(s)
        classes = detect_classes_dataframe(df)

        return learning_type, classes
    except Exception as e:
        # Fail-safe: handle any unexpected error gracefully
        logger.exception(f"Error in analyze_dataframe: {e}")
        return "Unsupervised", []


# ------------------------------------------------------------
# 2. JSON Analysis
# ------------------------------------------------------------
def analyze_json(data: Any) -> Tuple[str, List[Any]]:
    """
    Analyze a JSON-like object (dict or list) to infer learning type and extract class labels.

    Args:
        data (dict | list): Parsed JSON data.

    Returns:
        tuple:
            (str, list)
            - str: Detected learning type ("Supervised", "Semi-supervised", or "Unsupervised").
            - list: List of detected class labels (unique values).
    """
    try:
        # Recursively search for fields that may represent labels
        labels = find_labels_json_recursive(data)

        if labels:
            # Remove None and duplicates while preserving order
            classes = list(dict.fromkeys([l for l in labels if l is not None]))

            # Determine learning type:
            # If any None labels are present → Semi-supervised
            if any(l is None for l in labels):
                learning_type = "Semi-supervised"
            else:
                learning_type = "Supervised"
        else:
            # No labels found → Unsupervised learning
            learning_type = "Unsupervised"
            classes = []

        return learning_type, classes
    except Exception as e:
        # Fail-safe: avoid breaking the whole analysis if JSON is malformed
        logger.exception(f"[Error] in analyze_json(): {e}")
        return "Unsupervised", []


# ------------------------------------------------------------
# 3. Helper for Generic Use (Optional)
# ------------------------------------------------------------
def analyze_generic(data: Any) -> Tuple[str, List[Any]]:
    """
    General helper to analyze any data type (DataFrame or JSON-like).
    Automatically detects the correct analysis path.

    Args:
        data (pd.DataFrame | dict | list): The dataset to analyze.

    Returns:
        tuple:
            (str, list)
    """
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        return analyze_dataframe(data)
    elif isinstance(data, (dict, list)):
        return analyze_json(data)
    else:
        logger.warning("[Warning] Unsupported data type for analysis.")
        return "Unsupervised", []

