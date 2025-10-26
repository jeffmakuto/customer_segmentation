"""
Customer segmentation analysis using KMeans clustering.

This module loads the "Kenya supermarket" dataset (downloaded via Kaggle),
selects numeric features, finds a suitable number of clusters using the
silhouette score, fits a KMeans model, and writes several outputs to
``analysis_output/``:

- `cluster_summary.csv` - per-cluster count/mean/std for each numeric feature
- `cluster_sizes.csv` - number of observations in each cluster
- `cluster_profiles_mean.csv` - cluster mean values for numeric features
- `inertia_by_k.png` / `silhouette_by_k.png` - diagnostic plots for k selection
- `clusters_top2_features.png` - scatter plot of top two variance features
- `recommendations.txt` - simple, actionable recommendations for each cluster

The script is intended as a lightweight starting point for segmentation and
can be adapted to include more preprocessing, feature engineering, or other
clustering algorithms.

Usage:

        python clustering_analysis.py

Module-level assumptions:
        - The dataset file is present at the path used by the Kaggle download
            (default: ~/.cache/kagglehub/datasets/emmanuelkens/kenya-supermarkets-data/versions/2/Supermarket Data.xlsx).
        - The environment has the required packages installed (see `requirements.txt`).

"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    """Load data, perform KMeans clustering, and write outputs.

    This function performs the end-to-end pipeline for the simple clustering
    analysis used in the project. Steps include:

    1. Load the dataset from the expected local KaggleHub cache location.
    2. Select numeric features and perform basic missing-value handling.
    3. Standard scale features and evaluate k=2..8 using silhouette score.
    4. Fit a final KMeans model with the chosen k and save cluster outputs
       (summaries, cluster profiles, diagnostics plots, and recommendations).

    Args:
        None

    Returns:
        None

    Raises:
        SystemExit: if the dataset file is not found or no numeric columns are
            available for clustering.

    Notes:
        - The function writes files into an ``analysis_output/`` directory in
          the current working directory. Existing files with the same names
          will be overwritten.
        - The algorithm is intentionally simple (KMeans + silhouette) and is
          suitable for exploratory segmentation. For production use, more
          robust preprocessing, cross-validation, and stability checks are
          recommended.
    """
    # Path to the downloaded Excel file (adjust if you moved it)
    data_path = Path.home() / ".cache" / "kagglehub" / "datasets" / "emmanuelkens" / "kenya-supermarkets-data" / "versions" / "2" / "Supermarket Data.xlsx"
    if not data_path.exists():
        print("Dataset file not found:", data_path)
        sys.exit(1)

    print("Loading dataset from:", data_path)
    df = pd.read_excel(data_path)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    # Select numeric features for clustering
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] == 0:
        print("No numeric columns found for clustering. Exiting.")
        sys.exit(1)

    # Drop columns with too many missing values
    thresh = int(0.6 * len(numeric))
    numeric = numeric.dropna(axis=1, thresh=thresh)

    # Fill remaining missing values with median
    numeric = numeric.fillna(numeric.median())

    print("Numeric features used for clustering:", list(numeric.columns))

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(numeric)

    # Find best K using silhouette score for k=2..8
    best_k = None
    best_score = -1
    scores = {}
    inertias = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        try:
            score = silhouette_score(X, labels)
        except Exception:
            score = -1
        scores[k] = score
        inertias[k] = km.inertia_
        print(f"k={k}: silhouette={score:.4f}, inertia={km.inertia_:.2f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"Best k by silhouette: {best_k} (score={best_score:.4f})")

    # Ensure we have a valid k to pass to KMeans
    if best_k is None:
        print("Failed to determine a valid number of clusters. Exiting.")
        sys.exit(1)

    # Fit final model
    model = KMeans(n_clusters=int(best_k), random_state=42, n_init=20)
    labels = model.fit_predict(X)
    df['cluster'] = labels

    out_dir = Path.cwd() / 'analysis_output'
    out_dir.mkdir(exist_ok=True)

    # Save cluster summary
    summary = df.groupby('cluster')[numeric.columns].agg(['count', 'mean', 'std']).T
    summary.to_csv(out_dir / 'cluster_summary.csv')

    # Save cluster sizes
    cluster_sizes = df['cluster'].value_counts().sort_index()
    cluster_sizes.to_csv(out_dir / 'cluster_sizes.csv')
    print('Cluster sizes:\n', cluster_sizes)

    # Plot silhouette-like diagnostics: inertia and silhouette vs k
    plt.figure()
    ks = list(scores.keys())
    plt.plot(ks, [inertias[k] for k in ks], marker='o')
    plt.title('Elbow: Inertia by k')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.savefig(out_dir / 'inertia_by_k.png')

    plt.figure()
    plt.plot(ks, [scores[k] for k in ks], marker='o')
    plt.title('Silhouette by k')
    plt.xlabel('k')
    plt.ylabel('Silhouette score')
    plt.grid(True)
    plt.savefig(out_dir / 'silhouette_by_k.png')

    # Describe clusters
    cluster_profiles = df.groupby('cluster')[numeric.columns].mean()
    cluster_profiles.to_csv(out_dir / 'cluster_profiles_mean.csv')

    # Pick top 3 features by variance to plot
    variances = numeric.var().sort_values(ascending=False)
    top_features = list(variances.index[:3])
    print('Top features used for visualization:', top_features)

    # Scatter plot of first two top features
    if len(top_features) >= 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=numeric[top_features[0]], y=numeric[top_features[1]], hue=labels, palette='tab10')
        plt.title('Clusters by top features')
        plt.xlabel(top_features[0])
        plt.ylabel(top_features[1])
        plt.legend(title='cluster')
        plt.savefig(out_dir / 'clusters_top2_features.png')

    # Create simple actionable recommendations based on cluster means
    recommendations = []
    for c in cluster_profiles.index:
        profile = cluster_profiles.loc[c]
        high_spend_cols = [col for col in numeric.columns if profile[col] > numeric[col].mean()]
        low_spend_cols = [col for col in numeric.columns if profile[col] < numeric[col].mean()]
        rec = f"Cluster {c}: {int(cluster_sizes.get(c,0))} customers. "
        if 'Total' in numeric.columns:
            if profile['Total'] >= numeric['Total'].mean():
                rec += 'Higher-than-average spenders — consider VIP/loyalty offers. '
            else:
                rec += 'Lower-than-average spenders — consider promotional bundles. '
        # generic guidance
        if len(high_spend_cols) > 0:
            rec += f"High on: {', '.join(high_spend_cols[:3])}. "
        if len(low_spend_cols) > 0:
            rec += f"Low on: {', '.join(low_spend_cols[:3])}."
        recommendations.append(rec)

    with open(out_dir / 'recommendations.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(recommendations))

    print('\nRecommendations:')
    for r in recommendations:
        print('-', r)


if __name__ == '__main__':
    main()
