"""Data loading and basic cleaning utilities for the supermarket dataset.

This module provides functions to load the transaction-level data from the
downloaded Excel workbook (sheet "raw") and perform light cleaning such as
parsing dates and normalizing column names.

Functions
---------
load_raw(path=None)
    Load the "raw" sheet into a pandas DataFrame. If `path` is None, the
    function will look in the KaggleHub cache location used previously.
"""

from pathlib import Path
from typing import Optional
import pandas as pd


def default_data_path():
    """Return the default path to the downloaded Excel workbook."""
    return Path.home() / ".cache" / "kagglehub" / "datasets" / "emmanuelkens" / "kenya-supermarkets-data" / "versions" / "2" / "Supermarket Data.xlsx"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case and strip whitespace.

    This modifies the DataFrame column names in place and returns it.
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_').replace('.', '_') for c in df.columns]
    return df


def load_raw(path: Optional[str] = None) -> pd.DataFrame:
    """Load the transaction-level sheet named "raw" from the workbook.

    Args:
        path: Optional path to the Excel workbook. If None, the default cache
            path (created by the earlier download step) is used.

    Returns:
        A cleaned pandas DataFrame containing the raw transactions.
    """
    excel_path = Path(path) if path else default_data_path()
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    xls = pd.ExcelFile(excel_path)
    # Prefer a pre-engineered 'market' sheet (has extra categorical bins),
    # otherwise fall back to 'aggregate', then 'raw'. This mirrors best
    # practice for this workbook where 'market' contains helpful features.
    sheet_priority = ['market', 'aggregate', 'raw']
    found = None
    sheets_lower = {str(s).lower(): s for s in xls.sheet_names}
    for name in sheet_priority:
        if name in sheets_lower:
            found = sheets_lower[name]
            break
    if found is None:
        raise ValueError(f"Workbook does not contain any of the expected sheets {sheet_priority}. Found: {xls.sheet_names}")

    df = pd.read_excel(excel_path, sheet_name=found)
    df = _normalize_columns(df)

    # Try to combine date and time columns into a single datetime if present
    if 'date' in df.columns:
        try:
            # Some date columns may be strings; let pandas infer formats
            if 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
            else:
                df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
        except Exception:
            df['datetime'] = pd.to_datetime(df['date'], errors='coerce')

    return df
