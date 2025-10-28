## Customer Segmentation Analysis

This project builds a full customer segmentation workflow around the Kaggle **Kenyan Supermarket Transactions** dataset. A single notebook orchestrates loading the data, engineering transaction-level features, running clustering analysis, and exporting ready-to-use artefacts for reporting.

### Repository Structure

- `notebooks/segmentation_analysis.ipynb` – end-to-end notebook that engineers features, evaluates KMeans cluster counts, fits the final model, visualises segments, and saves outputs.
- `data_processing.py` – helpers to locate the workbook download and load the preferred `market` sheet with light cleaning (column normalisation, datetime assembly).
- `features.py` – feature engineering utilities for transactions and (optionally) store-level aggregates.
- `analysis_output/` – generated artefacts (CSV summaries, model binaries, charts) produced each time the notebook runs.
- `.venv/` – recommended virtual environment for local execution (excluded via `.gitignore`).
- `requirements.txt` – locked list of core packages required to execute the notebook.

### Getting Started

1. **Clone and enter the repo**
	```powershell
	git clone <your-fork-url>
	cd customer_segmentation
	```
2. **Create a virtual environment** (optional but recommended)
	```powershell
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1
	```
3. **Install dependencies**
	```powershell
	.\.venv\Scripts\python.exe -m pip install -r requirements.txt
	```
4. **Download the dataset** if you have not previously cached it via KaggleHub. The notebook expects the Excel file at:
	```text
	%USERPROFILE%\.cache\kagglehub\datasets\emmanuelkens\kenya-supermarkets-data\versions\2\Supermarket Data.xlsx
	```
	Adjust `data_processing.load_raw(path=...)` if your copy lives elsewhere.

### Running the Notebook

1. Launch VS Code or Jupyter Lab and open `notebooks/segmentation_analysis.ipynb`.
2. Select the `.venv` interpreter (or any environment containing the requirements) as the kernel.
3. Execute the notebook from top to bottom. Key stages include:
	- Data load and sanity checks.
	- Transaction feature engineering (`features.build_transaction_features`).
	- KMeans scaling, elbow/silhouette diagnostics, and best-`k` selection.
	- Visual summaries (histograms, diagnostic plots, cluster profiles).
	- Export of labelled transactions, cluster summaries, scaler/model artefacts, and PNG charts into `analysis_output/`.
4. Review the “Findings, insights, and next steps” markdown cell for business-ready takeaways.

### Outputs

Running the notebook refreshes these artefacts inside `analysis_output/`:

- `transactions_with_cluster_from_notebook.csv`
- `cluster_profiles_mean_from_notebook.csv`
- `cluster_sizes_from_notebook.csv`
- `cluster_small_rows.csv`
- `kmeans_model_from_notebook.joblib`
- `scaler_from_notebook.joblib`
- Diagnostic figures (`feature_distributions.png`, `k_selection_diagnostics.png`, `cluster_sizes_bar.png`, `cluster_scatter_total_items.png`).

Use the CSV exports for BI dashboards or further modelling, and the joblib files to reapply the trained scaler/model to future data.

### Extending the Analysis

- Add new feature engineering steps in `features.py` (e.g., basket composition ratios, customer loyalty flags) and rerun the notebook.
- Swap out KMeans for alternative clustering algorithms by editing the modelling sections.
- Leverage `build_store_aggregates` to produce store-level dashboards outside the notebook.

For questions or improvements, open an issue or submit a pull request.
