"""
intent_classifier.py
---------------------
Step 1 of the pipeline (NLP Layer): classify a customer query into
Fraud / Loan / KYC / Account Access, plus sentiment (6 real classes:
Anxious, Confused, Urgent, Neutral, Frustrated, Angry).

Both models are trained on real ground-truth labels from support_tickets.csv
(category and sentiment columns) — no heuristics.

Usage:
    python src/intent_classifier.py
    from intent_classifier import predict_intent
    predict_intent("I see a transaction I didn't make")
"""

from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data_loader import load_tickets

MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
INTENT_MODEL_PATH = MODEL_DIR / "intent_classifier.joblib"
SENTIMENT_MODEL_PATH = MODEL_DIR / "sentiment_classifier.joblib"


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, stop_words="english", min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def train():
    df = load_tickets()

    print("=" * 60)
    print("INTENT CLASSIFIER (category)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        df["query_text"], df["category"], test_size=0.2, random_state=42, stratify=df["category"]
    )
    intent_pipe = _build_pipeline()
    intent_pipe.fit(X_train, y_train)
    print(classification_report(y_test, intent_pipe.predict(X_test)))

    print("=" * 60)
    print("SENTIMENT CLASSIFIER (6-class)")
    print("=" * 60)
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        df["query_text"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
    )
    sentiment_pipe = _build_pipeline()
    sentiment_pipe.fit(Xs_train, ys_train)
    print(classification_report(ys_test, sentiment_pipe.predict(Xs_test)))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(intent_pipe, INTENT_MODEL_PATH)
    joblib.dump(sentiment_pipe, SENTIMENT_MODEL_PATH)
    print(f"\nModels saved to {MODEL_DIR}")

    # Caveat worth knowing: 200 tickets are built from ~66 repeated query
    # templates (~3x each), so held-out accuracy here is optimistic — it's
    # measuring template recognition more than generalization to novel
    # phrasing. Treat these scores as a pipeline sanity-check, not a
    # production accuracy estimate.
    return intent_pipe, sentiment_pipe


def load_models():
    if not (INTENT_MODEL_PATH.exists() and SENTIMENT_MODEL_PATH.exists()):
        print("No trained models found — training now...")
        return train()
    return joblib.load(INTENT_MODEL_PATH), joblib.load(SENTIMENT_MODEL_PATH)


def predict_intent(query: str, models=None) -> dict:
    intent_model, sentiment_model = models or load_models()
    intent = intent_model.predict([query])[0]
    intent_conf = float(intent_model.predict_proba([query]).max())
    sentiment = sentiment_model.predict([query])[0]
    sentiment_conf = float(sentiment_model.predict_proba([query]).max())
    return {
        "intent": intent,
        "confidence": round(intent_conf, 3),
        "sentiment": sentiment,
        "sentiment_confidence": round(sentiment_conf, 3),
    }


if __name__ == "__main__":
    models = train()
    sample = "I see a transaction of ₹10,000 I didn't make"
    print("\nSample prediction:")
    print(predict_intent(sample, models))
