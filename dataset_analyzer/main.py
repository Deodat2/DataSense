# ============================================================
# dataSense/main.py
# Interactive CLI for Dataset Analysis
# ============================================================

import argparse
import sys
import json
import logging
import warnings
import os

# Import the core analysis function from your package
from dataset_analyzer.core import dataset_identity, stop_spark_quietly

# ------------------------------------------------------------
# Logging configuration (configured here as the CLI entry point)
# ------------------------------------------------------------
# Set up a formatter for consistent output
LOG_FORMAT = "%(levelname)s: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DataSense")

# ------------------------------------------------------------
# Suppress unnecessary warnings
# ------------------------------------------------------------
# We can move this to a dedicated config file later, but keeping it here for now
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ------------------------------------------------------------
# Argument Parser Setup
# ------------------------------------------------------------
def create_parser():
    """Setup the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="DataSense: Automatic multimodal dataset analyzer.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "path",
        type=str,
        help="Path to the dataset folder or a single structured file."
    )

    parser.add_argument(
        "-o", "--output-json",
        type=str,
        default=None,
        help="Path to save the analysis report as a JSON file."
    )

    parser.add_argument(
        "-c", "--output-csv",
        type=str,
        default=None,
        help="Path to save the detailed per-file analysis as a CSV file."
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Increase output verbosity."
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable internal debug logging."
    )

    return parser


# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------
def main():
    parser = create_parser()
    args = parser.parse_args()

    # 1️⃣ Path Validation
    dataset_path = os.path.expanduser(args.path)
    if not os.path.exists(dataset_path):
        logger.error(f"❌ Path does not exist: {dataset_path}")
        sys.exit(1)
    # Note: We rely on core.py to handle file/folder logic for simplicity here

    # 2️⃣ Run dataset analysis
    logger.info(f"🔍 Starting analysis of: {dataset_path}")
    if args.debug:
        logger.setLevel(logging.DEBUG)  # Set logger level based on CLI argument

    try:
        # Pass the arguments directly to the core function
        result = dataset_identity(
            path_or_df=dataset_path,
            export_json=args.output_json,
            export_csv=args.output_csv,
            verbose=args.verbose,
            debug=args.debug  # Passes debug flag to core function's internal logic
        )

    except Exception as e:
        logger.exception("⚠️ An unexpected error occurred during analysis.")
        sys.exit(1)
    finally:
        # Crucial for stability: Stop Spark session if it was started
        stop_spark_quietly()

    # 3️⃣ Display results
    if result:
        logger.info("\n✅ Analysis complete!")
        logger.info("=" * 45)
        # We rely on dataset_identity to print the result if verbose=True

        logger.info("\n✨ Done! Use the generated JSON/CSV for detailed insights.")
        if args.verbose:
            print("\n--- Summary Report (Raw JSON) ---")
            print(json.dumps(result, indent=2, ensure_ascii=False))


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n❌ Analysis interrupted by user.")
        # Ensure Spark is stopped even on interrupt
        stop_spark_quietly()
        sys.exit(1)