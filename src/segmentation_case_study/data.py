"""Transaction loading, cleaning, and customer-level RFM feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "invoiceno": "InvoiceNo",
    "invoicedate": "InvoiceDate",
    "quantity": "Quantity",
    "unitprice": "UnitPrice",
    "customerid": "CustomerID",
}
RFM_COLUMNS = ["Recency", "Frequency", "Monetary"]


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel transaction file and normalize common column spellings."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError("Input must be a CSV or Excel workbook (.csv, .xlsx, .xls).")

    normalized = {str(column).strip().lower().replace(" ", "").replace("_", ""): column
                  for column in frame.columns}
    missing = [target for key, target in REQUIRED_COLUMNS.items() if key not in normalized]
    if missing:
        raise ValueError(
            "Input is missing required columns: "
            + ", ".join(missing)
            + f". Found: {', '.join(map(str, frame.columns))}"
        )
    frame = frame.rename(columns={normalized[key]: target for key, target in REQUIRED_COLUMNS.items()})
    return frame


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep identifiable, positive purchases and return a compact cleaning audit."""
    frame = raw.copy()
    rows_read = len(frame)
    frame["InvoiceNo"] = frame["InvoiceNo"].astype("string").str.strip()
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce", dayfirst=True)
    frame["Quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce")
    frame["UnitPrice"] = pd.to_numeric(frame["UnitPrice"], errors="coerce")
    customer = frame["CustomerID"].astype("string").str.strip()
    customer = customer.str.replace(r"\.0$", "", regex=True)
    frame["CustomerID"] = customer

    missing_customer = frame["CustomerID"].isna() | frame["CustomerID"].eq("")
    # Count removals in sequence so the audit categories are mutually exclusive.
    missing_rows = int(missing_customer.sum())
    frame = frame.loc[~missing_customer].copy()
    cancel_mask = frame["InvoiceNo"].str.upper().str.startswith("C", na=False)
    cancelled_rows = int(cancel_mask.sum())
    frame = frame.loc[~cancel_mask].copy()
    invalid_mask = (
        frame["InvoiceNo"].isna()
        | frame["InvoiceNo"].eq("")
        | frame["InvoiceDate"].isna()
        | frame["Quantity"].isna()
        | frame["UnitPrice"].isna()
        | (frame["Quantity"] <= 0)
        | (frame["UnitPrice"] <= 0)
    )
    invalid_rows = int(invalid_mask.sum())
    frame = frame.loc[~invalid_mask].copy()
    duplicate_mask = frame.duplicated(keep="first")
    duplicate_rows = int(duplicate_mask.sum())
    frame = frame.loc[~duplicate_mask].copy()
    frame["MonetaryLine"] = frame["Quantity"] * frame["UnitPrice"]
    frame = frame.loc[frame["MonetaryLine"] > 0].copy()

    audit = {
        "rows_read": int(rows_read),
        "rows_removed_missing_customer": missing_rows,
        "rows_removed_cancellations": cancelled_rows,
        "rows_removed_invalid_or_nonpositive": invalid_rows,
        "rows_removed_exact_duplicates": duplicate_rows,
        "rows_retained": int(len(frame)),
        "customers_retained": int(frame["CustomerID"].nunique()),
    }
    if frame.empty:
        raise ValueError("No valid purchase rows remain after cleaning.")
    return frame, audit


def build_rfm(
    transactions: pd.DataFrame, as_of: str | pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Aggregate purchase history into one positive RFM row per customer."""
    if transactions.empty:
        raise ValueError("Cannot build RFM features from an empty transaction table.")
    snapshot = (
        pd.Timestamp(as_of)
        if as_of is not None
        else transactions["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    )
    if snapshot.tz is not None:
        snapshot = snapshot.tz_localize(None)
    grouped = transactions.groupby("CustomerID", observed=True)
    rfm = grouped.agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("MonetaryLine", "sum"),
    )
    rfm["Recency"] = (snapshot.normalize() - rfm["LastPurchase"].dt.normalize()).dt.days
    rfm = rfm[["Recency", "Frequency", "Monetary"]].reset_index()
    if (rfm[RFM_COLUMNS] < 0).any().any():
        raise ValueError("RFM values must be non-negative for log1p transformation.")
    return rfm, snapshot.normalize()
