# ============================================================
# dataSense/main.py
# Interactive CLI for Dataset Analysis
# ============================================================

import os
import sys
import json
import logging
import warnings

from dataset_analyzer.core import dataset_identity

# ------------------------------------------------------------
# Suppress unnecessary warnings (PyTorch, Torchaudio, etc.)
# ------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DataSense")

# ------------------------------------------------------------
# Helper to validate dataset path
# ------------------------------------------------------------
def validate_dataset_path(path: str) -> str:
    """Validate folder path, exit if invalid."""
    if not path:
        logger.error("❌ No path entered. Exiting.")
        sys.exit(1)

    path = os.path.expanduser(path)
    if not os.path.exists(path):
        logger.error(f"❌ Path does not exist: {path}")
        sys.exit(1)
    if not os.path.isdir(path):
        logger.error(f"❌ Path is not a folder: {path}")
        sys.exit(1)
    return path

# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------
def main():
    logger.info("🔍 Welcome to DataSense Dataset Analyzer!")
    logger.info("="*45 + "\n")
    logger.info(
        "This tool helps you analyze any dataset folder —\n"
        "it automatically detects files (images, audio, text, etc.)\n"
        "and extracts meaningful insights.\n"
    )

    # 1️⃣ Ask user for dataset folder
    dataset_path = input("📂 Please enter the full path to your dataset folder: ").strip()
    dataset_path = validate_dataset_path(dataset_path)

    # 2️⃣ Run dataset analysis
    logger.info("\n🚀 Starting analysis...\n")
    try:
        result = dataset_identity(dataset_path, verbose=False)
    except Exception as e:
        logger.exception("⚠️ An error occurred during analysis:")
        sys.exit(1)

    # 3️⃣ Display results neatly
    logger.info("\n✅ Analysis complete!")
    logger.info("="*45)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("\n✨ Done! You can now use these insights in your workflow.\n")

# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n❌ Analysis interrupted by user.")
        sys.exit(1)