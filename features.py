"""Feature engineering helpers for transaction and store features.

This module produces a small, sensible set of features from the raw
transactions suitable for clustering. It keeps the logic simple and
documented so it can be extended.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of features (numeric) derived from transactions.

    Args:
        df: Raw transactions DataFrame (as returned by data_processing.load_raw).

    Returns:
        features: DataFrame where each row corresponds to the transaction in
            `df` (index aligned) and columns are numeric features ready for
            scaling and clustering.
    """
    df = df.copy()

    # Basic numeric features (ensure numeric dtypes)
    numeric_cols = ['no_of_items', 'variation', 'total', 'paid', 'change']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Datetime-derived features
    if 'datetime' in df.columns:
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['month'] = df['datetime'].dt.month

    # Frequency encoding for supermarket (simple categorical encoding)
    if 'supermarket' in df.columns:
        freq = df['supermarket'].fillna('UNKNOWN').astype(str).value_counts(normalize=True)
        df['supermarket_freq'] = df['supermarket'].fillna('UNKNOWN').astype(str).map(freq).fillna(0.0)

    # Payment method flags: try to detect main payment column 'type' or 'paid'
    if 'type' in df.columns:
        df['payment_type'] = df['type'].astype(str).str.lower()
        df['is_cash'] = df['payment_type'].str.contains('cash').astype(int)
        df['is_card'] = df['payment_type'].str.contains('card').astype(int)
        df['is_mpesa'] = df['payment_type'].str.contains('mpesa').astype(int)

    # Select features to return (only numeric)
    candidate_cols = ['no_of_items', 'variation', 'total', 'paid', 'change', 'hour', 'dayofweek', 'is_weekend', 'month', 'supermarket_freq', 'is_cash', 'is_card', 'is_mpesa']
    features = pd.DataFrame(index=df.index)
    for c in candidate_cols:
        if c in df.columns:
            features[c] = df[c]

    # Fill missing numeric with median (simple, robust)
    features = features.fillna(features.median())

    return features


def build_store_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level data to store-level summaries.

    Returns a DataFrame indexed by `supermarket` with aggregate metrics.
    """
    if 'supermarket' not in df.columns:
        raise ValueError('supermarket column is required for store aggregates')

    df2 = df.copy()
    # ensure numeric
    df2['total'] = pd.to_numeric(df2.get('total', 0), errors='coerce')
    grouped = df2.groupby('supermarket').agg(
        transactions=('total', 'count'),
        avg_total=('total', 'mean'),
        median_total=('total', 'median'),
        std_total=('total', 'std')
    ).fillna(0)

    return grouped
