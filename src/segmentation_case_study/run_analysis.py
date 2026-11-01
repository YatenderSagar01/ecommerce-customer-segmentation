"""Run the complete RFM customer segmentation case study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import build_rfm, clean_transactions, load_transactions
from .modeling import dbscan_epsilon_sensitivity, fit_models
from .personas import build_personas, profile_clusters
from .visualization import create_figures, reduce_dimensions


def run(
    input_path: Path,
    output_dir: Path = Path("reports"),
    as_of: str | None = None,
    max_k: int = 8,
    dbscan_eps: float = 0.8,
    min_samples: int = 8,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_transactions(input_path)
    transactions, audit = clean_transactions(raw)
    rfm, snapshot = build_rfm(transactions, as_of=as_of)
    clustering = fit_models(rfm, max_k=max_k, dbscan_eps=dbscan_eps, min_samples=min_samples)
    dbscan_sensitivity = dbscan_epsilon_sensitivity(
        clustering.scaled_features,
        min_samples=min_samples,
    )
    pca_coords, tsne_coords, explained = reduce_dimensions(clustering.scaled_features)

    assignments = clustering.assignments.copy()
    assignments["PCA1"] = pca_coords[:, 0]
    assignments["PCA2"] = pca_coords[:, 1]
    assignments["tSNE1"] = tsne_coords[:, 0]
    assignments["tSNE2"] = tsne_coords[:, 1]
    customer_segments = rfm.merge(assignments, on="CustomerID", validate="one_to_one")
    customer_segments.to_csv(output_dir / "customer_segments.csv", index=False)
    clustering.kmeans_scores.to_csv(output_dir / "kmeans_model_selection.csv", index=False)
    dbscan_sensitivity.to_csv(output_dir / "dbscan_epsilon_sensitivity.csv", index=False)

    profile_frames = [
        profile_clusters(rfm, clustering.assignments, "K-Means", "KMeansCluster"),
        profile_clusters(rfm, clustering.assignments, "DBSCAN", "DBSCANCluster"),
    ]
    profiles = pd.concat(profile_frames, ignore_index=True)
    profiles.to_csv(output_dir / "cluster_profiles.csv", index=False)
    personas = build_personas(rfm, profiles)
    personas.to_csv(output_dir / "marketing_personas.csv", index=False)

    create_figures(
        output_dir=output_dir,
        pca_coords=pca_coords,
        tsne_coords=tsne_coords,
        kmeans_labels=clustering.assignments["KMeansCluster"].to_numpy(),
        dbscan_labels=clustering.assignments["DBSCANCluster"].to_numpy(),
        kmeans_scores=clustering.kmeans_scores,
        explained_variance=explained,
        rfm=rfm,
        assignments=clustering.assignments,
        profiles=profiles,
    )

    best_k = int(clustering.kmeans_model.n_clusters)
    best_score = float(clustering.kmeans_scores.loc[
        clustering.kmeans_scores["k"] == best_k, "silhouette"
    ].iloc[0])
    summary = {
        "source_file": input_path.name,
        "snapshot_date": snapshot.date().isoformat(),
        "currency": "GBP",
        "cleaning": audit,
        "rfm_customer_count": int(len(rfm)),
        "kmeans": {
            "selected_k": best_k,
            "selection_metric": "silhouette score in standardized log-RFM space",
            "silhouette": best_score,
            "cluster_sizes": {str(int(k)): int(v) for k, v in
                              clustering.assignments["KMeansCluster"].value_counts().sort_index().items()},
        },
        "dbscan": {
            "eps": float(dbscan_eps),
            "min_samples": int(min_samples),
            "cluster_count_excluding_noise": int(
                clustering.assignments.loc[clustering.assignments["DBSCANCluster"] >= 0,
                                           "DBSCANCluster"].nunique()
            ),
            "noise_share": clustering.dbscan_noise_share,
            "silhouette_excluding_noise": clustering.dbscan_silhouette,
            "cluster_sizes_including_noise": {str(int(k)): int(v) for k, v in
                                               clustering.assignments["DBSCANCluster"].value_counts().sort_index().items()},
            "epsilon_sensitivity_file": "dbscan_epsilon_sensitivity.csv",
        },
        "pca": {"explained_variance_ratio": explained, "total_explained_variance": float(sum(explained))},
        "limitations": [
            "The transaction history is a single retailer's historical sample, not a current market view.",
            "Cluster labels describe relative purchase behavior and are not causal or immutable customer traits.",
            "t-SNE is a local-neighborhood visualization and should not be interpreted as preserving global distances.",
            "DBSCAN silhouette, when available, is calculated after excluding noise points.",
        ],
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_case_study_report(output_dir, summary, profiles, personas, dbscan_sensitivity)
    return summary


def write_case_study_report(
    output_dir: Path,
    summary: dict,
    profiles: pd.DataFrame,
    personas: pd.DataFrame,
    dbscan_sensitivity: pd.DataFrame,
) -> None:
    """Write a concise, data-driven narrative alongside machine-readable outputs."""
    cleaning = summary["cleaning"]
    kmeans = summary["kmeans"]
    dbscan = summary["dbscan"]
    pca = summary["pca"]
    lines = [
        "# E-Commerce Customer Segmentation: Case Study",
        "",
        "## Business question",
        "",
        "How can an online retailer use observed customer purchase behavior to prioritize retention, loyalty, and reactivation campaigns?",
        "",
        "## Executive summary",
        "",
        f"The cleaned UCI Online Retail extract contains **{cleaning['rows_retained']:,} purchase lines** from **{summary['rfm_customer_count']:,} customers**, with an RFM snapshot date of **{summary['snapshot_date']}**. K-Means selected **{kmeans['selected_k']} segments** from the tested range using the highest silhouette score (**{kmeans['silhouette']:.3f}**). DBSCAN found **{dbscan['cluster_count_excluding_noise']} dense segment(s)** at ε={dbscan['eps']:.2f} and marked **{dbscan['noise_share']:.1%}** of customers as noise. The first two PCA components explain **{pca['total_explained_variance']:.1%}** of standardized log-RFM variance.",
        "",
        "## Data preparation",
        "",
        "Source: [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online%2Bretail), covering transactions from December 2010 through December 2011. Recency is measured in days before the snapshot; frequency counts distinct purchase invoices; monetary value is observed spend in GBP.",
        "",
        "| Cleaning step | Rows |",
        "|---|---:|",
        f"| Source records | {cleaning['rows_read']:,} |",
        f"| Missing customer ID removed | {cleaning['rows_removed_missing_customer']:,} |",
        f"| Cancellation invoices removed | {cleaning['rows_removed_cancellations']:,} |",
        f"| Invalid or non-positive records removed | {cleaning['rows_removed_invalid_or_nonpositive']:,} |",
        f"| Exact duplicate rows removed | {cleaning['rows_removed_exact_duplicates']:,} |",
        f"| Purchase lines retained | {cleaning['rows_retained']:,} |",
        "",
        "Features were log1p-transformed and standardized before clustering. The default snapshot is one day after the final valid purchase date in the input.",
        "",
        "## K-Means personas",
        "",
        f"The selected K-Means model used k={kmeans['selected_k']}. Cluster IDs are arbitrary and should be interpreted through their RFM profiles.",
        "",
        "| Cluster | Customers | Share | Median recency | Median invoices | Median spend | Persona |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in personas[personas["method"] == "K-Means"].sort_values("cluster").iterrows():
        lines.append(
            f"| {int(row['cluster'])} | {int(row['customer_count']):,} | {row['share_pct']:.1f}% | "
            f"{row['median_recency']:.0f} days | {row['median_frequency']:.1f} | "
            f"£{row['median_monetary']:,.2f} | {row['persona']} |"
        )
    lines.extend(["", "Suggested campaign hypotheses:", ""])
    for _, row in personas[personas["method"] == "K-Means"].sort_values("cluster").iterrows():
        lines.append(f"- **{row['persona']} (cluster {int(row['cluster'])}):** {row['activation_idea']}")

    lines.extend([
        "",
        "## DBSCAN personas",
        "",
        f"With ε={dbscan['eps']:.2f} and min_samples={dbscan['min_samples']}, DBSCAN assigns dense points to clusters and leaves sparse observations as noise. Noise is not a customer persona and should not be targeted as one.",
        "",
        "| Cluster | Customers | Share | Median recency | Median invoices | Median spend | Persona |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ])
    for _, row in personas[personas["method"] == "DBSCAN"].sort_values("cluster").iterrows():
        lines.append(
            f"| {int(row['cluster'])} | {int(row['customer_count']):,} | {row['share_pct']:.1f}% | "
            f"{row['median_recency']:.0f} days | {row['median_frequency']:.1f} | "
            f"£{row['median_monetary']:,.2f} | {row['persona']} |"
        )
    lines.extend(["", "Suggested campaign hypotheses:", ""])
    for _, row in personas[personas["method"] == "DBSCAN"].sort_values("cluster").iterrows():
        lines.append(f"- **{row['persona']} (cluster {int(row['cluster'])}):** {row['activation_idea']}")
    lines.extend(["", "## DBSCAN parameter sensitivity", "",
                  "The epsilon sweep makes the density trade-off visible. Smaller neighborhoods produce more fragments and more noise; larger neighborhoods merge groups. Review the full table before applying DBSCAN settings to another retailer.",
                  "",
                  "| ε | Clusters | Noise | Smallest cluster | Largest cluster share |",
                  "|---:|---:|---:|---:|---:|"])
    for _, row in dbscan_sensitivity.iterrows():
        lines.append(
            f"| {row['eps']:.2f} | {int(row['cluster_count_excluding_noise'])} | "
            f"{row['noise_share_pct']:.1f}% | {int(row['smallest_cluster_size']):,} | "
            f"{row['largest_cluster_share_pct']:.1f}% |"
        )
    lines.extend([
        "",
        f"The analysis uses ε={dbscan['eps']:.2f}. At this setting it finds {dbscan['cluster_count_excluding_noise']} cluster(s), with {dbscan['noise_share']:.1%} of customers marked as noise.",
        "",
        "## Visualizations",
        "",
        "![K-Means model selection](figures/kmeans_model_selection.png)",
        "",
        "![PCA and t-SNE segment views](figures/pca_tsne_segments.png)",
        "",
        "![Median RFM profiles](figures/rfm_cluster_profiles.png)",
        "",
        "PCA preserves linear variance, while t-SNE is intended to show local neighborhoods and can distort global distances. Use the profiles and business experiments to judge usefulness, not apparent separation in a 2D map.",
        "",
        "## Recommended next steps",
        "",
        "1. Refresh RFM with recent transactions and evaluate whether segment memberships are stable over time.",
        "2. Run holdout-based tests for reactivation, loyalty, and cross-sell ideas; compare incremental margin and repeat purchase, not just opens or clicks.",
        "3. Suppress customers who do not respond and review opt-in, privacy, and contact-frequency rules before activation.",
        "",
        "## Limitations",
        "",
        "This is one retailer's historical sample with a finite observation window. The personas summarize relative purchase behavior; they are not causal, permanent, or validated response propensities. K-Means assigns every customer. DBSCAN's noise points are excluded when calculating its silhouette score.",
        "",
        "Dataset citation: Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).",
        "",
    ])
    (output_dir / "case_study_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Transaction file (.xlsx, .xls, or .csv).")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--as-of", default=None, help="Optional RFM snapshot date, e.g. 2011-12-10.")
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--dbscan-eps", type=float, default=0.35)
    parser.add_argument("--min-samples", type=int, default=8)
    args = parser.parse_args()
    summary = run(args.input, args.output_dir, args.as_of, args.max_k, args.dbscan_eps, args.min_samples)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
