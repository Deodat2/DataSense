# ============================================================
# dataset_analyzer/labels.py
# ------------------------------------------------------------
# Purpose:
#   Detect label columns (target variables) and infer
#   the learning type (supervised / semi-supervised / unsupervised)
#   in both tabular (DataFrame) and JSON data.
#
# Improvements over previous version:
#   - Configurable thresholds (cardinality ratio, missing ratio)
#   - Robust against empty DataFrames
#   - Smarter selection of label columns
#   - Clear and complete comments for educational purposes
# ============================================================

import pandas as pd
from typing import List, Any

# ------------------------------------------------------------
# 1️⃣ Detect label columns in DataFrame and infer learning type
# ------------------------------------------------------------
def detect_labels_dataframe_smart(
    df: pd.DataFrame,
    threshold: float = 0.2,
    missing_ratio_for_semi: float = 0.05,
    min_unique_abs: int = 2
)-> str:
    """
    Detect whether a DataFrame contains label-like columns
    and infer the learning type: supervised / semi-supervised / unsupervised.

    Args:
        df (pd.DataFrame): Input dataset.
        threshold (float): Maximum fraction of unique values allowed for a column
                           to be considered a potential label (default = 0.2).
        missing_ratio_for_semi (float): Fraction of missing values that makes
                                        a dataset semi-supervised (default = 0.05).
        min_unique_abs (int): Minimum number of unique values required
                              to consider a column as a candidate label.

    Returns:
        str: One of {"Supervised", "Semi-supervised", "Unsupervised"}.
    """

    # Handle empty DataFrame edge case
    if df.empty:
        return "Unsupervised"

    candidate_labels = []

    # ------------------------------------------------------------
    # Step 1: Identify candidate columns with low cardinality
    # ------------------------------------------------------------
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        # Skip columns with too few unique values (e.g., all same or empty)
        if nunique < min_unique_abs:
            continue

        ratio = nunique / len(df)
        if ratio < threshold:
            # Compute missing value ratio
            missing_ratio = df[col].isnull().mean()
            candidate_labels.append({
                "column": col,
                "unique_count": nunique,
                "ratio": ratio,
                "missing_ratio": missing_ratio
            })

    # If no candidate columns found → Unsupervised learning
    if not candidate_labels:
        return "Unsupervised"

    # ------------------------------------------------------------
    # Step 2: Heuristic to determine if dataset is semi-supervised
    # ------------------------------------------------------------
    for c in candidate_labels:
        if c["missing_ratio"] > missing_ratio_for_semi:
            return "Semi-supervised"

    # ------------------------------------------------------------
    # Step 3: Otherwise, dataset is supervised
    # ------------------------------------------------------------
    return "Supervised"


# ------------------------------------------------------------
# 2️⃣ Extract possible class names from a DataFrame
# ------------------------------------------------------------
def detect_classes_dataframe(
    df: pd.DataFrame,
    threshold: float = 0.2,
    min_unique_abs: int = 2
) -> List[Any]:
    """
    Extract possible class labels from a DataFrame
    by finding the most likely target column.

    Args:
        df (pd.DataFrame): Input dataset.
        threshold (float): Same as above, controls max uniqueness ratio.
        min_unique_abs (int): Minimum number of unique values to consider.

    Returns:
        list: Unique class names (or empty list if none found).
    """
    if df.empty:
        return []

    # Step 1: Find all label candidates
    candidate_labels = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique < min_unique_abs:
            continue
        ratio = nunique / len(df)
        if ratio < threshold:
            candidate_labels.append((col, ratio))

    # Step 2: If no candidate → return empty
    if not candidate_labels:
        return []

    # Step 3: Prefer column whose name suggests a label
    preferred_names = ["label", "target", "class", "classe", "category", "output", "y"]
    candidate_labels.sort(key=lambda x: x[1])  # smaller ratio first
    for col, _ in candidate_labels:
        if any(p in col.lower() for p in preferred_names):
            chosen_col = col
            break
    else:
        chosen_col = candidate_labels[0][0]

    # Step 4: Return list of unique class values
    return list(df[chosen_col].dropna().unique())


# ------------------------------------------------------------
# 3️⃣ Recursively find labels in a JSON structure
# ------------------------------------------------------------
def find_labels_json_recursive(obj: Any, candidate_keys: list[str] | None = None) -> list[Any]:
    """
    Recursively search for label-like keys inside a JSON object.

    Args:
        obj (dict | list | any): JSON-like object (nested dicts/lists).
        candidate_keys (list[str]): Keys to look for (case-insensitive).
                                    Default: ["label", "target", "class", "classe"].

    Returns:
        list: List of label values found anywhere in the structure.
    """
    if candidate_keys is None:
        candidate_keys = ["label", "target", "class", "classe"]

    labels = []

    # Case 1: dictionary
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in candidate_keys:
                labels.append(value)
            else:
                labels.extend(find_labels_json_recursive(value, candidate_keys))

    # Case 2: list
    elif isinstance(obj, list):
        for item in obj:
            labels.extend(find_labels_json_recursive(item, candidate_keys))

    # Case 3: other (ignore non-container types)
    return labels
