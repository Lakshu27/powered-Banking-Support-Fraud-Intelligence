"""
eda_utils.py
------------
Small, reusable EDA helpers shared across the notebooks so each notebook
stays focused on interpretation rather than boilerplate.
"""

import pandas as pd
import matplotlib.pyplot as plt


def basic_profile(df: pd.DataFrame, name: str = "dataset") -> None:
    """Print shape, dtypes, null counts and a head() preview."""
    print(f"--- {name} ---")
    print(f"shape: {df.shape}")
    print("\ndtypes:")
    print(df.dtypes)
    print("\nnull counts:")
    print(df.isnull().sum())
    print("\nduplicate rows:", df.duplicated().sum())
    print("\nhead:")
    print(df.head())


def plot_category_counts(df: pd.DataFrame, col: str, top_n: int = 20, title: str = None):
    counts = df[col].value_counts().head(top_n)
    ax = counts.plot(kind="bar", figsize=(8, 4))
    ax.set_title(title or f"Value counts: {col}")
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.show()
    return counts


def text_length_stats(df: pd.DataFrame, col: str) -> pd.Series:
    """Word-count stats for a free-text column — useful for tickets/queries."""
    lengths = df[col].astype(str).str.split().apply(len)
    print(lengths.describe())
    lengths.plot(kind="hist", bins=30, figsize=(8, 4), title=f"Word count: {col}")
    plt.xlabel("words")
    plt.tight_layout()
    plt.show()
    return lengths


def class_imbalance(df: pd.DataFrame, label_col: str) -> pd.Series:
    """Show class balance for a label column (e.g. fraud_label, category)."""
    counts = df[label_col].value_counts()
    pct = (counts / counts.sum() * 100).round(2)
    summary = pd.DataFrame({"count": counts, "pct": pct})
    print(summary)
    counts.plot(kind="pie", autopct="%1.1f%%", figsize=(5, 5), title=f"{label_col} balance")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
    return summary


def numeric_summary(df: pd.DataFrame, col: str, by: str = None):
    """Describe a numeric column, optionally grouped (e.g. amount by fraud_label)."""
    if by:
        print(df.groupby(by)[col].describe())
        df.boxplot(column=col, by=by, figsize=(7, 4))
        plt.title(f"{col} by {by}")
        plt.suptitle("")
        plt.tight_layout()
        plt.show()
    else:
        print(df[col].describe())
        df[col].plot(kind="hist", bins=40, figsize=(8, 4), title=f"Distribution: {col}")
        plt.tight_layout()
        plt.show()
