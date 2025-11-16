# ============================================================
# dataset_analyzer/core.py
# Automatic multimodal dataset analyzer (production-ready)
# ============================================================
"""
Professional, low-noise core for DataSense dataset analyzer.

Design choices:
- Use a module-scoped logger ("DataSense.core") instead of calling logging.basicConfig()
  so the application (main.py) controls global formatting/handlers.
- Initialize PySpark lazily and quietly (only when needed). When Spark is created we
  set Spark log level to ERROR and mute py4j/org logs to avoid JVM spam.
- Provide a `debug` flag on dataset_identity to temporarily raise verbosity for development.
- Keep function signatures backward compatible.
"""

from __future__ import annotations

import os
import json
import logging
import warnings
from collections import Counter
from typing import Union, Optional, Dict, Any, List

import pandas as pd

from .file_analysis import analyze_mixed_folder
from .content_analysis import analyze_dataframe
from .utils import fallback_summary

# -----------------------
# Module logger
# -----------------------
# Use a dedicated logger. Do NOT call basicConfig() here — main application should do that.
logger = logging.getLogger("DataSense.core")
# If no handler is configured by the application, this prevents "No handler" warnings.
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# -----------------------
# PySpark lazy initialization helper (quiet)
# -----------------------
_SPARK_INIT_DONE = False
_SPARK_AVAILABLE = False
_spark = None  # typing: ignore

def _init_spark_quietly() -> None:
    """
    Lazily initialize PySpark in quiet mode.
    This avoids starting the JVM / producing logs unless we actually need Spark.
    Sets module-level _spark and _SPARK_AVAILABLE.
    """
    global _SPARK_INIT_DONE, _SPARK_AVAILABLE, _spark

    if _SPARK_INIT_DONE:
        return

    _SPARK_INIT_DONE = True

    try:
        # Avoid passing problematic submit args that break some Windows setups.
        # Do not set PYSPARK_SUBMIT_ARGS globally here.
        # Try importing PySpark — may raise if Java/GW not available.
        from pyspark.sql import SparkSession

        # Create a small, quiet SparkSession; disable console progress UI
        _spark = (
            SparkSession.builder
            .appName("DataSense-DatasetIdentity")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.sql.repl.eagerEval.enabled", "false")
            .getOrCreate()
        )

        # Silence Spark and JVM related logs (py4j, org)
        try:
            _spark.sparkContext.setLogLevel("ERROR")
        except Exception:
            # be permissive -- sometimes sparkContext may not be ready
            pass

        logging.getLogger("py4j").setLevel(logging.ERROR)
        logging.getLogger("org").setLevel(logging.ERROR)

        # Filter noisy warnings from lower-level libs while allowing critical ones
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        _SPARK_AVAILABLE = True
        logger.info("✅ PySpark initialized (quiet mode).")
    except Exception as e:
        # Spark not available or failed to initialize — fallback to pandas-only
        _SPARK_AVAILABLE = False
        _spark = None
        logger.debug("PySpark not available: %s", e)


# Expose basic spark availability variable for other modules (read-only)
def _spark_session():
    """Return the active Spark session or None. Initializes lazily."""
    if not _SPARK_INIT_DONE:
        _init_spark_quietly()
    return _spark


# Threshold (bytes) to trigger Spark for very large structured files
SPARK_THRESHOLD = 100_000_000  # 100 MB


# ============================================================
# Main entry: dataset_identity
# ============================================================
def dataset_identity(
    path_or_df: Union[str, pd.DataFrame],
    export_json: Optional[str] = None,
    export_csv: Optional[str] = None,
    verbose: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Analyze a dataset (folder, file, or DataFrame) to produce a structured report.

    Args:
        path_or_df: Path to a folder, a single file, or a Pandas DataFrame.
        export_json: Optional path to write the JSON report.
        export_csv: Optional path to write per-file details CSV.
        verbose: If True, prints a compact JSON to stdout.
        debug: If True, raises logger level to DEBUG for this call (temporary).

    Returns:
        dict: analysis report
    """
    # Temporarily adjust logger level if debug requested
    original_level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)

    try:
        # Initialize (quiet) Spark only when we need it
        spark = _spark_session()
        use_spark_global = bool(_SPARK_AVAILABLE)

        recognized_files = {
            "csv": [],
            "json": [],
            "images": [],
            "audio": [],
            "text": [],
            "unknown": []
        }
        non_recognized_files: List[Union[str, Dict]] = []
        learning_types_global: List[str] = []
        file_summary: List[Dict[str, Any]] = []

        def record_file_summary(file_path: str, ftype: str, learning_type: str, classes: Any):
            file_summary.append({
                "file_path": file_path,
                "file_type": ftype,
                "learning_type": learning_type,
                "classes": classes
            })

        # -----------------------
        # Case: DataFrame provided directly
        # -----------------------
        if isinstance(path_or_df, pd.DataFrame):
            try:
                lt, classes = analyze_dataframe(path_or_df)
                learning_types_global.append(lt)
                record_file_summary("DataFrame", "tabular", lt, classes)
            except Exception as e:
                logger.exception("Error analyzing DataFrame: %s", e)
                return {}

        # -----------------------
        # Case: Path (file or folder)
        # -----------------------
        elif isinstance(path_or_df, str):
            if not os.path.exists(path_or_df):
                logger.warning("Path does not exist: %s", path_or_df)
                return {}

            # Folder
            if os.path.isdir(path_or_df):
                try:
                    analyze_mixed_folder(
                        path_or_df,
                        recognized_files,
                        non_recognized_files,
                        learning_types_global,
                        record_file_summary
                    )
                except Exception as e:
                    logger.exception("Error analyzing folder %s: %s", path_or_df, e)
                    return {}

            # Single file
            elif os.path.isfile(path_or_df):
                folder = os.path.dirname(path_or_df)

                analyze_mixed_folder(
                    folder,
                    recognized_files,
                    non_recognized_files,
                    learning_types_global,
                    record_file_summary
                )

            else:
                logger.warning("Unsupported path: %s", path_or_df)
                return {}

        # -----------------------
        # Build report
        # -----------------------
        total_files = sum(len(v) for v in recognized_files.values()) + len(non_recognized_files)
        global_learning_type = (
            Counter(learning_types_global).most_common(1)[0][0]
            if learning_types_global else "Undetermined"
        )
        valid_counts = {k: len(v) for k, v in recognized_files.items() if len(v) > 0}
        global_structure = max(valid_counts, key=valid_counts.get) if valid_counts else "Undetermined"

        report = {
            "analyzed_source": str(path_or_df) if not isinstance(path_or_df, pd.DataFrame) else "DataFrame",
            "use_spark": bool(use_spark_global),
            "total_files": total_files,
            "recognized_files": {k: len(v) for k, v in recognized_files.items()},
            "non_recognized_files": [fallback_summary(f) for f in non_recognized_files],
            "global_learning_type": global_learning_type,
            "global_structure": global_structure,
            # "files_detail": file_summary,
        }

        # -----------------------
        # Optional exports
        # -----------------------
        if export_json:
            try:
                with open(export_json, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                logger.info("Report exported as JSON: %s", export_json)
            except Exception:
                logger.exception("Failed to export JSON report to %s", export_json)

        if export_csv and file_summary:
            try:
                pd.DataFrame(file_summary).to_csv(export_csv, index=False)
                logger.info("File details exported as CSV: %s", export_csv)
            except Exception:
                logger.exception("Failed to export CSV details to %s", export_csv)

        # -----------------------
        # Print result (if requested)
        # -----------------------
        if verbose:
            # Keep printing compact so main.py formatting remains simple
            print(json.dumps(report, indent=2, ensure_ascii=False))

        return report

    finally:
        # restore original logger level if we changed it
        if debug:
            logger.setLevel(original_level)


# ============================================================
# Cleanup helper: stop Spark session (optional)
# ============================================================
def stop_spark_quietly() -> None:
    """
    If Spark was initialized by this module, stop it quietly.
    Call at program shutdown if needed.
    """
    spark = _spark_session()
    if spark is not None:
        try:
            spark.stop()
            logger.info("Spark session stopped.")
        except Exception:
            logger.warning("Failed to stop Spark session cleanly.")
