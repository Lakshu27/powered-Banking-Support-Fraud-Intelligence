"""
db_setup.py
------------
Loads the CSV/JSON dataset into a normalized SQLite database with indexes,
per the project brief's "SQL Database Practices" guideline:
  - Normalize tables (avoid redundancy)
  - Use indexes on commonly-queried columns
  - Consistent, descriptive naming

Run once (or whenever the raw data changes) to (re)build data/processed/banking.db:
    python src/db_setup.py

Everything downstream can then query this DB instead of reading CSVs
directly — see db_loader.py for drop-in replacements for data_loader.py's
functions.
"""

import sqlite3
from pathlib import Path

from data_loader import load_tickets, load_transactions, load_qa_pairs

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "banking.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id               TEXT PRIMARY KEY,
    date_created            TEXT NOT NULL,
    customer_id             TEXT NOT NULL,
    channel                 TEXT NOT NULL,
    category                TEXT NOT NULL,
    query_text              TEXT NOT NULL,
    sentiment               TEXT NOT NULL,
    risk_level              TEXT NOT NULL,
    resolution_text         TEXT NOT NULL,
    resolution_time_minutes INTEGER NOT NULL,
    resolved_by             TEXT NOT NULL,
    customer_satisfaction   INTEGER NOT NULL,
    escalated               INTEGER NOT NULL
);
-- sub_category deliberately excluded: confirmed corrupted/truncated in EDA
-- (see notebooks/01_eda_support_tickets.ipynb) — normalizing garbage data
-- in is still garbage data, so it's dropped rather than stored.

CREATE INDEX IF NOT EXISTS idx_tickets_category   ON tickets(category);
CREATE INDEX IF NOT EXISTS idx_tickets_risk_level ON tickets(risk_level);
CREATE INDEX IF NOT EXISTS idx_tickets_sentiment  ON tickets(sentiment);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id     TEXT PRIMARY KEY,
    account_id          TEXT NOT NULL,
    ts                  TEXT NOT NULL,
    amount_inr          REAL NOT NULL,
    merchant_name       TEXT NOT NULL,
    merchant_category   TEXT NOT NULL,
    transaction_type    TEXT NOT NULL,
    city                TEXT NOT NULL,
    hour_of_day         INTEGER NOT NULL,
    day_of_week         TEXT NOT NULL,
    is_international    INTEGER NOT NULL,
    velocity_flag       INTEGER NOT NULL,
    geo_anomaly_flag    INTEGER NOT NULL,
    high_amount_flag    INTEGER NOT NULL,
    fraud_label         INTEGER NOT NULL,
    fraud_reason        TEXT
);

CREATE INDEX IF NOT EXISTS idx_txn_fraud_label ON transactions(fraud_label);
CREATE INDEX IF NOT EXISTS idx_txn_city        ON transactions(city);
CREATE INDEX IF NOT EXISTS idx_txn_type        ON transactions(transaction_type);

CREATE TABLE IF NOT EXISTS qa_pairs (
    qa_id            TEXT PRIMARY KEY,
    category         TEXT NOT NULL,
    question         TEXT NOT NULL,
    answer           TEXT NOT NULL,
    policy_ref       TEXT NOT NULL,
    risk_level       TEXT NOT NULL,
    suggested_action TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_qa_category   ON qa_pairs(category);
CREATE INDEX IF NOT EXISTS idx_qa_risk_level ON qa_pairs(risk_level);
"""


def build_database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()  # start clean so re-running this script doesn't duplicate rows
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)

    tickets = load_tickets()
    tickets_out = tickets[[
        "ticket_id", "date_created", "customer_id", "channel", "category",
        "query_text", "sentiment", "risk_level", "resolution_text",
        "resolution_time_minutes", "resolved_by", "customer_satisfaction", "escalated",
    ]].copy()
    tickets_out["date_created"] = tickets_out["date_created"].astype(str)
    tickets_out["escalated"] = tickets_out["escalated"].astype(int)
    tickets_out.to_sql("tickets", conn, if_exists="append", index=False)

    tx = load_transactions()
    tx_out = tx.rename(columns={"timestamp": "ts", "transaction_id": "transaction_id"}).copy()
    tx_out["ts"] = tx_out["ts"].astype(str)
    for c in ["is_international", "velocity_flag", "geo_anomaly_flag", "high_amount_flag"]:
        tx_out[c] = tx_out[c].astype(int)
    tx_out = tx_out[[
        "transaction_id", "account_id", "ts", "amount_inr", "merchant_name",
        "merchant_category", "transaction_type", "city", "hour_of_day", "day_of_week",
        "is_international", "velocity_flag", "geo_anomaly_flag", "high_amount_flag",
        "fraud_label", "fraud_reason",
    ]]
    tx_out.to_sql("transactions", conn, if_exists="append", index=False)

    qa = load_qa_pairs().rename(columns={"id": "qa_id"})
    qa[["qa_id", "category", "question", "answer", "policy_ref", "risk_level", "suggested_action"]]\
        .to_sql("qa_pairs", conn, if_exists="append", index=False)

    conn.commit()

    # sanity check: row counts + confirm indexes exist
    cur = conn.cursor()
    for table in ("tickets", "transactions", "qa_pairs"):
        n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {n} rows")
    indexes = cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'").fetchall()
    print(f"Indexes created: {[i[0] for i in indexes]}")

    conn.close()
    print(f"\nDatabase built at {DB_PATH}")


if __name__ == "__main__":
    build_database()
