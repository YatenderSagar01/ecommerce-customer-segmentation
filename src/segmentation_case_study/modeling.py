"""Feature transforms and customer clustering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .data import RFM_COLUMNS


@dataclass
class ClusteringResult:
    assignments: pd.DataFrame
    scaled_features: np.ndarray
    scaler: StandardScaler
    kmeans_model: KMeans
    kmeans_scores: pd.DataFrame
    dbscan_model: DBSCAN
    dbscan_silhouette: float | None
    dbscan_noise_share: float


def fit_models(
    rfm: pd.DataFrame,
    max_k: int = 8,
    dbscan_eps: float = 0.35,
    min_samples: int = 8,
    random_state: int = 42,
) -> ClusteringResult:
    if len(rfm) < 3:
        raise ValueError("At least three customers are required for clustering.")
    if max_k < 2:
        raise ValueError("--max-k must be at least 2.")

    log_features = np.log1p(rfm[RFM_COLUMNS].astype(float))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(log_features)
    upper_k = min(max_k, len(rfm) - 1)
    score_rows: list[dict[str, float | int]] = []
    fitted: dict[int, KMeans] = {}
    for k in range(2, upper_k + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(scaled)
        score = silhouette_score(scaled, labels)
        score_rows.append({"k": k, "silhouette": float(score), "inertia": float(model.inertia_)})
        fitted[k] = model
    scores = pd.DataFrame(score_rows)
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    kmeans = fitted[best_k]
    kmeans_labels = kmeans.labels_

    dbscan = DBSCAN(eps=dbscan_eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(scaled)
    non_noise = dbscan_labels != -1
    non_noise_labels = dbscan_labels[non_noise]
    cluster_count = len(np.unique(non_noise_labels))
    dbscan_silhouette = None
    if cluster_count >= 2 and non_noise.sum() > cluster_count:
        dbscan_silhouette = float(silhouette_score(scaled[non_noise], non_noise_labels))

    assignments = pd.DataFrame(
        {"CustomerID": rfm["CustomerID"].astype(str), "KMeansCluster": kmeans_labels,
         "DBSCANCluster": dbscan_labels}
    )
    return ClusteringResult(
        assignments=assignments,
        scaled_features=scaled,
        scaler=scaler,
        kmeans_model=kmeans,
        kmeans_scores=scores,
        dbscan_model=dbscan,
        dbscan_silhouette=dbscan_silhouette,
        dbscan_noise_share=float((dbscan_labels == -1).mean()),
    )


def dbscan_epsilon_sensitivity(
    scaled_features: np.ndarray,
    min_samples: int = 8,
    eps_values: tuple[float, ...] = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
) -> pd.DataFrame:
    """Summarize how DBSCAN density settings change cluster sizes and noise."""
    rows = []
    for eps in eps_values:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled_features)
        counts = pd.Series(labels).value_counts()
        cluster_sizes = counts[counts.index != -1]
        rows.append({
            "eps": eps,
            "min_samples": min_samples,
            "cluster_count_excluding_noise": int(len(cluster_sizes)),
            "noise_share_pct": float((labels == -1).mean() * 100),
            "smallest_cluster_size": int(cluster_sizes.min()) if len(cluster_sizes) else 0,
            "largest_cluster_share_pct": float(cluster_sizes.max() / len(labels) * 100) if len(cluster_sizes) else 0.0,
        })
    return pd.DataFrame(rows)
