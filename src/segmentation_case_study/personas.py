"""Cluster profile summaries and behavior-based marketing personas."""

from __future__ import annotations

import pandas as pd

METRICS = ["Recency", "Frequency", "Monetary"]

ACTIVATION = {
    "Champions": "Reward advocacy with early access or a referral offer; protect margin with recognition before discounting.",
    "Promising regulars": "Use a relevant cross-sell and a light loyalty nudge to build the next purchase habit.",
    "New or occasional buyers": "Send a helpful post-purchase journey and a small second-order incentive with a clear expiry.",
    "At-risk high value": "Prioritize a personal win-back message, replenishment reminder, or service check before a broad discount.",
    "Frequent low-basket buyers": "Test bundles, threshold-based free delivery, or complementary products to grow basket value.",
    "Big spenders": "Offer curated premium recommendations and attentive service; measure margin as well as revenue.",
    "Needs reactivation": "Run a low-cost reactivation test and suppress further spend if the customer stays inactive.",
    "Steady customers": "Maintain useful product updates and test a modest loyalty benefit against a no-offer holdout.",
    "Unassigned (noise)": "Review individually or with a separate rule-based audience; avoid treating sparse behavior as a stable persona.",
}


def profile_clusters(rfm: pd.DataFrame, labels: pd.DataFrame, method: str, label_column: str) -> pd.DataFrame:
    joined = rfm.merge(labels[["CustomerID", label_column]], on="CustomerID", validate="one_to_one")
    profiles = (
        joined.groupby(label_column, dropna=False)
        .agg(customer_count=("CustomerID", "size"), **{f"median_{m.lower()}": (m, "median") for m in METRICS})
        .reset_index()
        .rename(columns={label_column: "cluster"})
    )
    profiles["method"] = method
    profiles["share_pct"] = profiles["customer_count"] / len(joined) * 100
    return profiles[["method", "cluster", "customer_count", "share_pct",
                     "median_recency", "median_frequency", "median_monetary"]]


def _relative_band(value: float, lower: float, upper: float) -> str:
    if value <= lower:
        return "low"
    if value >= upper:
        return "high"
    return "mid"


def _choose_persona(recency: str, frequency: str, monetary: str) -> str:
    if recency == "low" and frequency == "high" and monetary == "high":
        return "Champions"
    if recency == "low" and frequency in {"mid", "high"}:
        return "Promising regulars"
    if recency == "low" and frequency == "low":
        return "New or occasional buyers"
    if recency == "high" and frequency in {"mid", "high"} and monetary in {"mid", "high"}:
        return "At-risk high value"
    if frequency == "high" and monetary == "low":
        return "Frequent low-basket buyers"
    if monetary == "high":
        return "Big spenders"
    if recency == "high":
        return "Needs reactivation"
    return "Steady customers"


def build_personas(rfm: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    lower = rfm[METRICS].quantile(0.33)
    upper = rfm[METRICS].quantile(0.67)
    rows = []
    for profile in profiles.to_dict("records"):
        cluster = int(profile["cluster"])
        if profile["method"] == "DBSCAN" and cluster == -1:
            persona = "Unassigned (noise)"
            r_band = f"{profile['median_recency']:.0f} days"
            f_band = f"{profile['median_frequency']:.1f} orders"
            m_band = f"£{profile['median_monetary']:,.2f}"
        else:
            r_band = _relative_band(profile["median_recency"], lower["Recency"], upper["Recency"])
            f_band = _relative_band(profile["median_frequency"], lower["Frequency"], upper["Frequency"])
            m_band = _relative_band(profile["median_monetary"], lower["Monetary"], upper["Monetary"])
            persona = _choose_persona(r_band, f_band, m_band)
        rows.append({
            **profile,
            "persona": persona,
            "recency_vs_customer_base": r_band,
            "frequency_vs_customer_base": f_band,
            "monetary_vs_customer_base": m_band,
            "activation_idea": ACTIVATION[persona],
        })
    result = pd.DataFrame(rows)
    return result.sort_values(["method", "cluster"]).reset_index(drop=True)
