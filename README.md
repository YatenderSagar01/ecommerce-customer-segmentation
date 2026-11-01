# E-Commerce Customer Segmentation

An end-to-end RFM segmentation case study using K-Means and DBSCAN on the UCI Online Retail transactions. The workflow cleans transaction data, builds customer-level Recency, Frequency, and Monetary features, scales their skew-reduced values, compares clustering methods, reduces the feature space with PCA, visualizes customer relationships with t-SNE, and turns cluster profiles into testable marketing personas.

## Case study question

How can an online retailer group customers by their observed purchase behavior and use those groups to design more relevant retention, loyalty, and reactivation campaigns?

## Data

This project uses the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online%2Bretail), a transactional dataset covering a UK-based non-store retailer from 1 December 2010 through 9 December 2011. It contains invoice, product, quantity, timestamp, unit price, customer, and country fields. UCI reports 541,909 records and makes the dataset available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please cite: Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33.

The raw workbook is included at `data/raw/Online Retail.xlsx` so the analysis can run immediately after dependency installation. To replace it with a fresh UCI copy, run:

```bash
python -m pip install -r requirements.txt
python -m segmentation_case_study.download_data
```

You can also pass your own CSV or Excel transaction extract with the same column names.

## Run the analysis

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m segmentation_case_study.download_data
python -m segmentation_case_study.run_analysis --input data/raw/Online Retail.xlsx
```

To run against a CSV:

```bash
python -m segmentation_case_study.run_analysis --input path/to/transactions.csv
```

Useful options:

```text
--as-of YYYY-MM-DD     Set the RFM snapshot date. Default: one day after the
                       latest valid purchase in the input.
--max-k 8              Largest K-Means cluster count considered (2 through max-k).
--dbscan-eps 0.35      DBSCAN neighborhood radius in standardized log-RFM space.
--min-samples 8        DBSCAN minimum neighborhood size.
--output-dir reports   Destination for tables, summary JSON, and figures.
```

The UCI workbook is about 23 MB; the t-SNE map is calculated over the available customer records and may take a little time.

## Method

1. **Prepare transactions.** Normalize input headers, parse dates and customer IDs, remove exact duplicate rows, drop records without a customer ID, exclude cancellation invoices (invoice numbers beginning with `C`), and retain positive quantities and unit prices. Monetary value is quantity multiplied by unit price, in GBP.
2. **Build RFM.** For each customer, recency is the number of days from their last included purchase to the snapshot date; frequency is the number of distinct included invoices; monetary value is total included spend. The default snapshot is one day after the latest included invoice date.
3. **Transform and scale.** Apply `log1p` to each positive RFM feature, then standardize each feature. This reduces the influence of the long-tailed frequency and spend distributions while keeping the three dimensions on comparable scales.
4. **Fit K-Means.** Search from 2 through `--max-k`; select the candidate with the highest silhouette score in scaled RFM space. K-Means is deterministic with a fixed seed and multiple initializations.
5. **Fit DBSCAN.** Use the supplied `eps` and `min_samples` values in the same scaled space. DBSCAN can mark sparse observations as noise (-1), so the report gives the noise share and only calculates a silhouette score when the non-noise result supports it.
6. **Reduce dimensions.** Fit 2-component PCA on scaled RFM for a variance-aware view. Fit t-SNE on the same features for a nonlinear neighborhood visualization. Both plots color customers by each model's labels.
7. **Profile and activate.** Compare each cluster's median RFM values to overall medians, assign a descriptive persona, and pair it with a practical campaign idea. These are hypotheses for testing, not fixed customer traits.

## Outputs

The analysis writes these files under `reports/`:

- `customer_segments.csv`: one row per customer with RFM, both model labels, PCA coordinates, and t-SNE coordinates.
- `cluster_profiles.csv`: customer count, share, and median RFM metrics by method and cluster.
- `marketing_personas.csv`: behavior-based persona summary and suggested activation for every cluster (including DBSCAN noise).
- `analysis_summary.json`: chosen K, silhouette scores where defined, DBSCAN noise rate, PCA variance, and run parameters.
- `case_study_report.md`: data-driven narrative with cluster profiles, personas, DBSCAN sensitivity, recommendations, and figures.
- `dbscan_epsilon_sensitivity.csv`: density, noise, and cluster-size trade-offs across a small epsilon grid.
- `figures/kmeans_model_selection.png`: silhouette and inertia by candidate K.
- `figures/pca_segments.png`: K-Means and DBSCAN in PCA space.
- `figures/tsne_segments.png`: K-Means and DBSCAN in t-SNE space.
- `figures/rfm_cluster_profiles.png`: median RFM profiles by model and cluster.

## Results snapshot

On the UCI extract, cleaning retains 392,692 purchase lines from 4,338 identified customers. Silhouette selection chooses two K-Means groups: a recent, higher-frequency, higher-spend group and a less-recent, mostly one-invoice group. The selected DBSCAN setting (ε=0.35, min_samples=8) finds three dense groups and labels 4.6% as noise. The first two PCA components explain 93.9% of standardized log-RFM variance. The [generated case study report](reports/case_study_report.md) contains the full profiles, persona actions, parameter sensitivity table, and figures.

## Reading the results

Use cluster profiles to understand differences in purchase behavior, not just the visual maps. PCA preserves linear variance; t-SNE emphasizes local neighborhoods and can distort global distances. Neither plot proves that clusters are naturally separated. K-Means always assigns a cluster, while DBSCAN may leave customers unassigned as noise. Silhouette values are diagnostics, not business outcomes, and are not directly comparable across algorithms when DBSCAN has noise.

The source covers a limited historical period and a single retailer. Recency is measured relative to that period's final observed date, so personas should be refreshed with current transaction data before campaigns are launched. Validate campaign ideas with holdouts, monitor incremental revenue and unsubscribe rates, and review privacy and consent requirements for the intended use.

## Project structure

```text
.
├── data/
│   └── raw/                         # included UCI workbook
├── reports/                         # analysis tables, summary, and figures
├── src/segmentation_case_study/
│   ├── data.py                      # loading, cleaning, and RFM features
│   ├── download_data.py             # UCI dataset downloader
│   ├── modeling.py                  # transforms and clustering
│   ├── personas.py                  # cluster profiles and activation ideas
│   ├── run_analysis.py              # command-line pipeline
│   └── visualization.py             # PCA, t-SNE, and profile plots
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Reproducibility

The random seed is fixed at 42 for K-Means and t-SNE. Dependency ranges are in `requirements.txt`. DBSCAN's `eps` is an explicit modeling choice and should be sensitivity-tested for a new dataset.
