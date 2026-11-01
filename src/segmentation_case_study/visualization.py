"""Publication-ready visual summaries for segmentation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_dimensions(scaled: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, list[float]]:
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(scaled)
    perplexity = min(30.0, max(2.0, (len(scaled) - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        max_iter=1000,
    )
    tsne_coords = tsne.fit_transform(scaled)
    return pca_coords, tsne_coords, pca.explained_variance_ratio_.tolist()


def _scatter(ax: plt.Axes, coords: np.ndarray, labels: np.ndarray, title: str) -> None:
    unique = sorted(np.unique(labels))
    palette = plt.get_cmap("tab10")
    for i, label in enumerate(unique):
        mask = labels == label
        name = "Noise" if label == -1 else f"Cluster {label}"
        ax.scatter(coords[mask, 0], coords[mask, 1], s=17, alpha=0.68,
                   color="#9aa0a6" if label == -1 else palette(i % 10), label=name,
                   edgecolors="none")
    ax.set_title(title, fontsize=12, weight="bold")
    prefix = "t-SNE" if "t-SNE" in title else "PCA"
    ax.set_xlabel(f"{prefix} 1")
    ax.set_ylabel(f"{prefix} 2")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8, loc="best")


def create_figures(
    output_dir: Path,
    pca_coords: np.ndarray,
    tsne_coords: np.ndarray,
    kmeans_labels: np.ndarray,
    dbscan_labels: np.ndarray,
    kmeans_scores: pd.DataFrame,
    explained_variance: list[float],
    rfm: pd.DataFrame,
    assignments: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(kmeans_scores["k"], kmeans_scores["inertia"], marker="o", color="#146c94")
    axes[0].set(title="K-Means inertia", xlabel="Number of clusters (k)", ylabel="Within-cluster sum of squares")
    axes[1].plot(kmeans_scores["k"], kmeans_scores["silhouette"], marker="o", color="#e07a5f")
    axes[1].set(title="K-Means silhouette", xlabel="Number of clusters (k)", ylabel="Silhouette score")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("K-Means model selection", weight="bold")
    fig.tight_layout()
    fig.savefig(figures / "kmeans_model_selection.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _scatter(axes[0, 0], pca_coords, kmeans_labels, f"K-Means — PCA ({sum(explained_variance):.1%} variance)")
    _scatter(axes[0, 1], pca_coords, dbscan_labels, f"DBSCAN — PCA ({sum(explained_variance):.1%} variance)")
    _scatter(axes[1, 0], tsne_coords, kmeans_labels, "K-Means — t-SNE")
    _scatter(axes[1, 1], tsne_coords, dbscan_labels, "DBSCAN — t-SNE")
    fig.suptitle("Customer segments in two-dimensional views", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(figures / "pca_tsne_segments.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Add per-model PCA and t-SNE views under stable filenames referenced in the README.
    for method, coords, labels, filename, title in [
        ("K-Means", pca_coords, kmeans_labels, "pca_segments.png", "PCA"),
        ("DBSCAN", pca_coords, dbscan_labels, "pca_dbscan_segments.png", "PCA"),
        ("K-Means", tsne_coords, kmeans_labels, "tsne_segments.png", "t-SNE"),
        ("DBSCAN", tsne_coords, dbscan_labels, "tsne_dbscan_segments.png", "t-SNE"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        _scatter(ax, coords, labels, f"{method} customer segments — {title}")
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    plot_data = rfm.merge(assignments[["CustomerID", "KMeansCluster", "DBSCANCluster"]], on="CustomerID")
    methods = [("KMeansCluster", "K-Means"), ("DBSCANCluster", "DBSCAN")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (column, method) in enumerate(methods):
        for col, metric in enumerate(["Recency", "Frequency", "Monetary"]):
            ax = axes[row, col]
            grouped = plot_data.groupby(column)[metric].median().sort_index()
            colors = ["#9aa0a6" if idx == -1 else plt.get_cmap("tab10")(i % 10)
                      for i, idx in enumerate(grouped.index)]
            ax.bar([str(i) for i in grouped.index], grouped.values, color=colors)
            ax.axhline(rfm[metric].median(), color="#333333", linestyle="--", linewidth=1, label="Customer median")
            ax.set_title(f"{method}: median {metric}")
            ax.set_xlabel("Cluster (−1 = noise)")
            ax.grid(axis="y", alpha=0.18)
            if metric == "Monetary":
                ax.set_ylabel("GBP")
            else:
                ax.set_ylabel("Days" if metric == "Recency" else "Invoices")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Median customer behavior by cluster", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(figures / "rfm_cluster_profiles.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
