# DataSense Dataset Analyzer v0.5

Automatic multimodal dataset analyzer (tabular, images, audio, text).  
This tool provides **implicit automated analysis** of datasets, producing a structured report for ML tasks.

## Features

- Analyze entire folders, single files, or pandas DataFrames.
- Detect file types: CSV, JSON, images, audio, text, and unknown.
- Automatic detection of global dataset structure and learning type (classification, regression, etc.).
- Lazy PySpark support for very large datasets (>100MB) to speed up analysis.
- Detailed per-file summaries with optional CSV export.
- Simple ASCII progress bars for tracking analysis.
- Robust reading for CSV/Excel/JSON files with encoding fallbacks.
- Modular design for easy extension (add new file types or analysis methods).

## Installation

```bash
git clone https://github.com/Deodat2/DataSense.git
cd DataSense
pip install -r requirements.txt
python main.py
```

## Output result Example

```bash
{
    "analyzed_source": "my_dataset/",
    "use_spark": true,
    "total_files": 123,
    "recognized_files": {
        "csv": 40,
        "json": 10,
        "images": 60,
        "audio": 5,
        "text": 5,
        "unknown": 3
    },
    "non_recognized_files": [
        {"path": "unknown_file.xyz", "mime": "unknown"}
    ],
    "global_learning_type": "Classification",
    "global_structure": "images"
}
```