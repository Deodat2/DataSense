# dataset_analyzer/file_analysis.py
"""
File Analysis Module - Clean Reimplementation (Option A)

Features:
- Single fast pre-scan to group files by type
- Per-type processing with an ASCII progress bar
- Lazy model imports (images/audio)
- Robust safe-readers for CSV/Excel/JSON
- Keeps existing callbacks and structures (record_file_summary, recognized_files, etc.)
"""

import os
import json
import logging
import pandas as pd
from typing import List, Dict, Callable, Any

from .content_analysis import analyze_dataframe, analyze_json
from .utils import fallback_summary

logger = logging.getLogger(__name__)

# File type groups
EXT_GROUPS = {
    "csv": [".csv", ".xls", ".xlsx"],
    "json": [".json"],
    "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp"],
    "audio": [".wav", ".mp3", ".flac", ".m4a"],
    "text": [".txt"],
}

# -------------------------
# Helpers
# -------------------------
def _detect_type(ext: str) -> str:
    ext = ext.lower()
    for t, exts in EXT_GROUPS.items():
        if ext in exts:
            return t
    return "unknown"

def _progress_bar(current: int, total: int, prefix: str = ""):
    if total <= 0:
        return

    # Longueur de la barre
    bar_len = 40

    # Calcul de la partie remplie
    filled_len = int(bar_len * current / total)

    # Partie verte =
    GREEN = "\033[92m"
    RESET = "\033[0m"

    bar = GREEN + "=" * filled_len + RESET + " " * (bar_len - filled_len)

    # Affichage inline
    print(f"\r{prefix} [{bar}] {current}/{total}", end="", flush=True)

    # Quand terminé, nouvelle ligne
    if current == total:
        print()

# -------------------------
# Safe readers
# -------------------------
def _safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        logger.warning(f"Encoding issue for {path}, trying latin-1.")
        return pd.read_csv(path, encoding="latin-1")
    except Exception:
        logger.exception(f"Failed to read CSV: {path}")
        raise

def _safe_read_excel(path: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception:
        logger.exception(f"Failed to read Excel: {path}")
        raise

def _safe_read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return json.load(f)
    except Exception:
        logger.exception(f"Failed to parse JSON: {path}")
        raise

# -------------------------
# Phase 1: scan and group
# -------------------------
def scan_dataset(path: str) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {t: [] for t in EXT_GROUPS.keys()}
    grouped["unknown"] = []
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            t = _detect_type(ext)
            grouped.setdefault(t, []).append(fp)
    logger.info("Dataset scan completed.")
    return grouped

# -------------------------
# Phase 2: per-type processing functions
# -------------------------
def _process_csv_group(filepaths: List[str], recognized_files, non_recognized_files, learning_types_global, record_file_summary):
    total = len(filepaths)
    for i, p in enumerate(filepaths, 1):
        try:
            df = _safe_read_csv(p) if p.lower().endswith(".csv") else _safe_read_excel(p)
            lt, classes = analyze_dataframe(df)
            recognized_files["csv"].append(p)
            learning_types_global.append(lt)
            record_file_summary(p, "csv", lt, classes)
        except Exception:
            non_recognized_files.append(p)
            logger.exception("CSV processing error: %s", p)
        _progress_bar(i, total, prefix="CSV")

def _process_json_group(filepaths: List[str], recognized_files, non_recognized_files, learning_types_global, record_file_summary):
    total = len(filepaths)
    for i, p in enumerate(filepaths, 1):
        try:
            data = _safe_read_json(p)
            lt, classes = analyze_json(data)
            recognized_files["json"].append(p)
            learning_types_global.append(lt)
            record_file_summary(p, "json", lt, classes)
        except Exception:
            non_recognized_files.append(p)
            logger.exception("JSON processing error: %s", p)
        _progress_bar(i, total, prefix="JSON")

def _process_images_group(
        filepaths: List[str],
        recognized_files,
        non_recognized_files,
        learning_types_global,
        record_file_summary
):
    total = len(filepaths)
    from .models.image_model import detect_image_class
    for i, p in enumerate(filepaths, 1):
        try:
            label = detect_image_class(p) or "Indeterminate"
            recognized_files["images"].append(p)
            record_file_summary(p, "images", "Not applicable", label)
        except Exception:
            non_recognized_files.append(p)
            logger.exception("Image processing error: %s", p)
        _progress_bar(i, total, prefix="Images")

def _process_audio_group(
        filepaths: List[str],
        recognized_files,
        non_recognized_files,
        learning_types_global,
        record_file_summary
):
    total = len(filepaths)
    from .models.audio_model import detect_audio_class
    for i, p in enumerate(filepaths, 1):
        try:
            label = detect_audio_class(p) or "Indeterminate"
            recognized_files["audio"].append(p)
            record_file_summary(p, "audio", "Not applicable", label)
        except Exception:
            non_recognized_files.append(p)
            logger.exception("Audio processing error: %s", p)
        _progress_bar(i, total, prefix="Audio")

def _process_text_group(filepaths: List[str], recognized_files, non_recognized_files, learning_types_global, record_file_summary):
    total = len(filepaths)
    for i, p in enumerate(filepaths, 1):
        try:
            recognized_files["text"].append(p)
            record_file_summary(p, "text", "Not applicable", "Not applicable")
        except Exception:
            non_recognized_files.append(p)
            logger.exception("Text processing error: %s", p)
        _progress_bar(i, total, prefix="Text")

def _process_unknown_group(filepaths: List[str], recognized_files, non_recognized_files, learning_types_global, record_file_summary):
    total = len(filepaths)
    for i, p in enumerate(filepaths, 1):
        try:
            meta = fallback_summary(p)
            record_file_summary(p, meta.get("mime", "unknown"), "Unknown", [])
            non_recognized_files.append(p)
        except Exception:
            non_recognized_files.append(p)
            logger.exception("Unknown file processing error: %s", p)
        _progress_bar(i, total, prefix="Unknown")

# Mapping
PROCESSORS = {
    "csv": _process_csv_group,
    "json": _process_json_group,
    "images": _process_images_group,
    "audio": _process_audio_group,
    "text": _process_text_group,
    "unknown": _process_unknown_group,
}

# -------------------------
# Public API: analyze_mixed_folder
# -------------------------
def analyze_mixed_folder(
        path: str,
        recognized_files: Dict[str, List[str]],
        non_recognized_files: List[str],
        learning_types_global: List[str],
        record_file_summary: Callable[[str, str, str, Any], None],
):
    if not os.path.isdir(path):
        logger.warning("Path is not a directory: %s", path)
        return {"recognized": recognized_files, "non_recognized": non_recognized_files, "learning_types": learning_types_global}

    # 1) scan
    groups = scan_dataset(path)

    # 2) summary log
    logger.info("Found file groups:")
    for k, v in groups.items():
        if v:
            logger.info("  - %s: %d", k, len(v))

    # 3) process each group
    for ftype, lst in groups.items():
        if not lst:
            continue
        logger.info("▶ Processing %s (%d files)", ftype, len(lst))
        processor = PROCESSORS.get(ftype)
        if processor:
            processor(lst, recognized_files, non_recognized_files, learning_types_global, record_file_summary)
        else:
            logger.warning("No processor for type: %s", ftype)

    logger.info("Folder analysis completed for: %s", path)
    return {
        "recognized": recognized_files,
        "non_recognized": non_recognized_files,
        "learning_types": learning_types_global,
    }
