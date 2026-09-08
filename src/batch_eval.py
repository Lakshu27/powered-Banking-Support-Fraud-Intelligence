"""
batch_eval.py
---------------
Runs every question in qa_pairs.json through the full pipeline and prints
predicted vs. reference side by side, so you can actually check whether
each query gets a distinct, sensible answer instead of the same generic
fallback every time.

Usage: python src/batch_eval.py
"""

from data_loader import load_qa_pairs
from pipeline import run_pipeline

def main():
    qa = load_qa_pairs()
    for _, row in qa.iterrows():
        result = run_pipeline(row["question"])
        print("=" * 90)
        print(f"[{row['id']}] {row['question']}")
        print(f"  predicted: intent={result['intent']} sentiment={result['sentiment']} risk={result['risk']}")
        print(f"  expected:  category={row['category']} risk={row['risk_level']}")
        print(f"  generated response: {result['response'][:200]}")
        print(f"  reference answer:   {row['answer'][:200]}")
        print(f"  suggested action:   {result['action']}")

if __name__ == "__main__":
    main()
