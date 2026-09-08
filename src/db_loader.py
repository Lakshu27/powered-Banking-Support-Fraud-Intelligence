"""
db_loader.py
-------------
SQLite-backed equivalents of data_loader.py's functions — same output
schema (columns, dtypes, normalized category names), so anything importing
from data_loader can switch to db_loader with a one-line import change.

Requires data/processed/banking.db to exist — run `python src/db_setup.py`
first (or this will do it automatically on first use).
"""

from pathlib import Path
import sqlite3
import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "banking.db"


def _ensure_db():
    if not DB_PATH.exists():
        from db_setup import build_database
        build_database()


def load_tickets() -> pd.DataFrame:
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM tickets", conn, parse_dates=["date_created"])
    conn.close()
    df["escalated"] = df["escalated"].astype(bool)
    return df


def load_transactions() -> pd.DataFrame:
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM transactions", conn, parse_dates=["ts"])
    conn.close()
    df = df.rename(columns={"ts": "timestamp"})
    for c in ["is_international", "velocity_flag", "geo_anomaly_flag", "high_amount_flag"]:
        df[c] = df[c].astype(bool)
    return df


def load_qa_pairs() -> pd.DataFrame:
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM qa_pairs", conn)
    conn.close()
    return df.rename(columns={"qa_id": "id"})


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Escape hatch for ad-hoc SQL — e.g. filtered/aggregated queries that
    don't need a full table load. Example:
        query("SELECT category, COUNT(*) FROM tickets GROUP BY category")
    """
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df


if __name__ == "__main__":
    print(load_tickets().shape)
    print(load_transactions().shape)
    print(load_qa_pairs().shape)
    print("\nExample ad-hoc query:")
    print(query("SELECT category, COUNT(*) as n, AVG(resolution_time_minutes) as avg_mins FROM tickets GROUP BY category"))
