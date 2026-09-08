"""
data_loader.py
---------------
Loads the real datasets for the AI-Powered Banking Support & Fraud
Intelligence System (NLP + RAG) capstone project.

Files expected under data/raw/ (see data/raw/DATASET_README.md for the
full schema the provider documented):

    support_tickets.csv   -> 200 tickets, columns incl. category, query_text,
                              sentiment, risk_level, resolution_text
    transactions.csv      -> 2,000 transactions, incl. amount_inr, fraud_label,
                              velocity_flag, geo_anomaly_flag, high_amount_flag
    qa_pairs.json          -> 20 curated Q&A pairs for RAG evaluation
    policies/*.txt         -> 4 policy documents (fraud, KYC, loan, refund)
"""

from pathlib import Path
import json
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

TICKETS_FILE = RAW_DIR / "support_tickets.csv"
TRANSACTIONS_FILE = RAW_DIR / "transactions.csv"
QA_PAIRS_FILE = RAW_DIR / "qa_pairs.json"
POLICIES_DIR = RAW_DIR / "policies"

# The tickets file uses "Fraud/Unauthorized"; qa_pairs.json uses "Fraud".
# Normalize both to one label set so a model trained on one can be
# evaluated against the other.
CATEGORY_NORMALIZE = {
    "fraud/unauthorized": "Fraud",
    "fraud": "Fraud",
    "loan": "Loan",
    "kyc": "KYC",
    "account access": "Account Access",
}


def _normalize_category(series: pd.Series) -> pd.Series:
    return series.str.strip().str.lower().map(CATEGORY_NORMALIZE).fillna(series)


def load_tickets(path: Path = TICKETS_FILE) -> pd.DataFrame:
    """Load the customer support tickets dataset (200 rows)."""
    if not path.exists():
        raise FileNotFoundError(f"Tickets file not found at {path}.")
    df = pd.read_csv(path, parse_dates=["date_created"])
    df.columns = [c.strip().lower() for c in df.columns]
    df["category"] = _normalize_category(df["category"])
    df["escalated"] = df["escalated"].str.strip().str.lower().eq("yes")
    return df


def load_transactions(path: Path = TRANSACTIONS_FILE) -> pd.DataFrame:
    """Load the structured transactions dataset (2,000 rows)."""
    if not path.exists():
        raise FileNotFoundError(f"Transactions file not found at {path}.")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.columns = [c.strip().lower() for c in df.columns]
    for flag_col in ("is_international", "velocity_flag", "geo_anomaly_flag", "high_amount_flag"):
        if flag_col in df.columns:
            df[flag_col] = df[flag_col].str.strip().str.lower().eq("yes")
    return df


def load_qa_pairs(path: Path = QA_PAIRS_FILE) -> pd.DataFrame:
    """Load the curated Q&A pairs used to evaluate RAG retrieval quality."""
    if not path.exists():
        raise FileNotFoundError(f"QA pairs file not found at {path}.")
    data = json.loads(path.read_text())
    df = pd.DataFrame(data)
    df["category"] = _normalize_category(df["category"])
    return df


def list_policy_docs(folder: Path = POLICIES_DIR) -> list[Path]:
    """Return paths of all policy documents available for RAG ingestion."""
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in (".txt", ".pdf", ".md")
    )
