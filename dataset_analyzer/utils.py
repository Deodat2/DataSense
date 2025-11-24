# ============================================================
# dataset_analyzer/utils.py
# ------------------------------------------------------------
# Utility functions for generic file information extraction.
# Provides a safe fallback summary when a file cannot be deeply analyzed.
# ------------------------------------------------------------
# Improvements over the original:
# - Uses pathlib for cleaner, cross-platform paths
# - Adds error handling (file not found, permission denied, etc.)
# - Adds optional SHA-1 hash computation
# - Adds optional "human-readable" file size
# - Adds last modification timestamp
# - Option to use python-magic for MIME detection (if available)
# ============================================================

from pathlib import Path
import mimetypes
import logging
import hashlib
from typing import Dict, Any, Union

# Try to import python-magic for better MIME detection (optional dependency)
try:
    import magic  # pip install python-magic
    _HAS_MAGIC = True
except (ImportError, ModuleNotFoundError):
    _HAS_MAGIC = False

# Configure logger (used instead of print)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Helper: Convert bytes into a human-readable format
# ------------------------------------------------------------
def human_readable_size(num_bytes: int) -> str:
    """
    Convert a number of bytes into a human-readable string.
    Example: 1536000 -> '1.5 MB'
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


# ------------------------------------------------------------
# Main function: fallback_summary
# ------------------------------------------------------------
def fallback_summary(
        file_path: Union[str, Path],
        compute_sha1: bool = False,
        use_magic: bool = False
) -> dict[str, any]:
    """
    Safely return basic metadata about a file.

    Args:
        file_path (str): Path to the file.
        compute_sha1 (bool): If True, computes the file's SHA-1 hash.
                             This can be slow for very large files.
        use_magic (bool): If True and python-magic is available,
                          detects MIME type based on file content
                          instead of just file extension.

    Returns:
        dict: Dictionary containing:
              - path (str): File path
              - size_bytes (int): File size in bytes
              - size_human (str): Human-readable file size (e.g., "2.3 MB")
              - mime (str): Detected MIME type or "unknown"
              - last_modified (float): Unix timestamp of last modification
              - is_dir (bool): Whether the path is a directory
              - sha1 (str, optional): SHA1 checksum if requested
              - error (str, optional): Present if an error occurred
    """
    p = Path(file_path)
    info: Dict[str, Any] = {"path": str(p)}

    # --------------------------------------------------------
    # Step 1: Gather basic file system information
    # --------------------------------------------------------
    try:
        stat = p.stat()  # Get file metadata (size, modification time, etc.)
        info["size_bytes"] = stat.st_size
        info["size_human"] = human_readable_size(stat.st_size)
        info["last_modified"] = stat.st_mtime
        info["is_dir"] = p.is_dir()
    except FileNotFoundError:
        # File does not exist
        logger.warning(f"File not found: {file_path}")
        return {"path": str(p), "error": "file_not_found"}
    except PermissionError:
        # File exists but cannot be accessed
        logger.warning(f"Permission denied for file: {file_path}")
        return {"path": str(p), "error": "permission_denied"}
    except Exception as e:
        # Any other unexpected OS-level error
        logger.error(f"Error reading file info ({file_path}): {e}")
        return {"path": str(p), "error": str(e)}

    # --------------------------------------------------------
    # Step 2: Detect MIME type
    # --------------------------------------------------------
    mime_type = None

    # Option 1: Use python-magic (content-based detection)
    if use_magic and _HAS_MAGIC:
        try:
            mime_type = magic.from_file(str(p), mime=True)
        except Exception as e:
            logger.debug(f"python-magic failed on {file_path}: {e}")

    # Option 2: Fallback to mimetypes (extension-based detection)
    if mime_type is None:
        mime_type, _ = mimetypes.guess_type(str(p))

    info["mime"] = mime_type or "unknown"

    # --------------------------------------------------------
    # Step 3: Optional SHA-1 hash computation
    # --------------------------------------------------------
    if compute_sha1 and p.is_file():
        try:
            h = hashlib.sha1()
            # Read in chunks (memory-efficient for large files)
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            info["sha1"] = h.hexdigest()
        except Exception as e:
            logger.debug(f"Could not compute SHA1 for {file_path}: {e}")
            info["sha1"] = None

    # --------------------------------------------------------
    # Step 4: Return the result dictionary
    # --------------------------------------------------------
    return info


# Example usage (for debugging only)
if __name__ == "__main__":
    test_path = "example.txt"
    summary = fallback_summary(test_path, compute_sha1=True, use_magic=True)
    print(summary)
