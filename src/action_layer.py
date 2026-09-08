"""
action_layer.py
-----------------
Step 5 of the pipeline: turn (intent, risk level) into a concrete suggested
action. The table below is seeded from the suggested_action values in the
real qa_pairs.json ground truth (grouped by category + risk_level), not
invented — check data/raw/qa_pairs.json to see the source examples.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from data_loader import load_qa_pairs

DEFAULT_ACTION = "Route to a human agent for manual assessment."


def build_action_table() -> dict:
    """Derives one representative action per (category, risk_level) pair
    from qa_pairs.json's suggested_action field."""
    qa = load_qa_pairs()
    table = {}
    for _, row in qa.iterrows():
        key = (row["category"], row["risk_level"])
        table.setdefault(key, row["suggested_action"])
    return table


ACTION_TABLE = build_action_table()


def suggest_action(intent: str, risk: str) -> str:
    return ACTION_TABLE.get((intent, risk), DEFAULT_ACTION)


if __name__ == "__main__":
    for k, v in ACTION_TABLE.items():
        print(k, "->", v)
    print("\nfallback (Loan, High):", suggest_action("Loan", "High"))
