"""
risk_classifier.py
--------------------
Step 4 of the pipeline: assign a risk level (High / Medium / Low).

Two real, trained models (no rule-guessing — both trained on ground truth
in the provided dataset):

  1. Ticket-text risk model: query_text -> risk_level (from support_tickets.csv).
     Used when we only have the customer's message (the live-query flow).

  2. Transaction fraud model: transaction features -> fraud_label
     (from transactions.csv), then fraud probability is banded into
     Low/Medium/High. Used for batch-scoring transactions or when a
     transaction record is available.

EDA finding worth knowing (see notebooks/02_eda_transactions.ipynb):
`is_international` and `geo_anomaly_flag` are perfectly correlated with
fraud_label in this dataset (both always Yes for fraud, always No for
legitimate). That's a strong, realistic signal by design — these are the
kind of rule-engine flags real fraud systems use upstream of an ML model —
but it does mean this particular synthetic dataset makes the transaction
model's job easy. Don't expect the same separation on live data.
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from data_loader import load_tickets, load_transactions

MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
TICKET_RISK_MODEL_PATH = MODEL_DIR / "ticket_risk_model.joblib"
FRAUD_MODEL_PATH = MODEL_DIR / "transaction_fraud_model.joblib"

RISK_ORDER = ["Low", "Medium", "High"]
NUMERIC_FEATURES = ["amount_inr", "hour_of_day"]
CATEGORICAL_FEATURES = ["merchant_category", "transaction_type", "city", "day_of_week"]
BOOL_FEATURES = ["is_international", "velocity_flag", "geo_anomaly_flag", "high_amount_flag"]


def train_ticket_risk_model():
    """Trained on query_text + category jointly, not text alone. Category
    is a strong predictor of risk_level here (Fraud tickets skew High,
    Account Access skews Low — see notebooks/01_eda_support_tickets.ipynb)
    but the text-only version couldn't use that signal explicitly. Adding
    it as a feature raised end-to-end risk match on qa_pairs.json from
    75% to 85% (see src/batch_eval.py) — including fixing the one case
    that mattered most: a CVV-phishing call going from a Medium
    misclassification to the correct High."""
    df = load_tickets()
    X = df[["query_text", "category"]]
    y = df["risk_level"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, stop_words="english"), "query_text"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["category"]),
    ])
    pipe = Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    print("Ticket-text + category risk model:")
    print(classification_report(y_test, pipe.predict(X_test)))

    # refit on all 200 rows for the shipped model (train/test split above is
    # just for the printed accuracy estimate)
    pipe.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, TICKET_RISK_MODEL_PATH)
    return pipe


def train_transaction_fraud_model():
    df = load_transactions()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOL_FEATURES].copy()
    for c in BOOL_FEATURES:
        X[c] = X[c].astype(int)
    y = df["fraud_label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ], remainder="passthrough")

    pipe = Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    print("\nTransaction fraud model:")
    print(classification_report(y_test, pipe.predict(X_test)))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, FRAUD_MODEL_PATH)
    return pipe


def train():
    ticket_model = train_ticket_risk_model()
    fraud_model = train_transaction_fraud_model()
    return ticket_model, fraud_model


def load_ticket_risk_model():
    if not TICKET_RISK_MODEL_PATH.exists():
        return train_ticket_risk_model()
    return joblib.load(TICKET_RISK_MODEL_PATH)


def load_fraud_model():
    if not FRAUD_MODEL_PATH.exists():
        return train_transaction_fraud_model()
    return joblib.load(FRAUD_MODEL_PATH)


def classify_risk_from_text(query: str, category: str = None) -> str:
    """Live-query flow: only the customer's message is available.
    `category` should be the predicted intent (from intent_classifier.py) —
    passing it makes a real difference here (75% -> 85% match on
    qa_pairs.json) since category correlates strongly with risk_level.
    If not supplied, this predicts it internally so the function still
    works standalone."""
    import pandas as pd
    if category is None:
        from intent_classifier import predict_intent
        category = predict_intent(query)["intent"]
    model = load_ticket_risk_model()
    X = pd.DataFrame([{"query_text": query, "category": category}])
    return model.predict(X)[0]


def classify_risk_from_transaction(transaction: dict) -> str:
    """Batch/known-transaction flow: full transaction record available.
    `transaction` needs the columns in NUMERIC_FEATURES + CATEGORICAL_FEATURES
    + BOOL_FEATURES (booleans as True/False)."""
    model = load_fraud_model()
    row = {**transaction}
    for c in BOOL_FEATURES:
        row[c] = int(bool(row.get(c, False)))
    X = pd.DataFrame([row])[NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOL_FEATURES]
    prob = model.predict_proba(X)[0][1]
    if prob >= 0.7:
        return "High"
    if prob >= 0.3:
        return "Medium"
    return "Low"


def classify_risk(intent: str, query: str = "", transaction: dict = None) -> str:
    """Main entry point used by the rest of the pipeline. Uses the
    transaction fraud model if a transaction record is supplied, otherwise
    falls back to the ticket-text risk model (live chat flow)."""
    if transaction:
        return classify_risk_from_transaction(transaction)
    if query:
        return classify_risk_from_text(query)
    return "Low"


if __name__ == "__main__":
    train()

    print("\nSample ticket-text prediction:")
    print(classify_risk_from_text("I see a transaction of ₹25,000 I didn't make"))

    print("\nSample transaction prediction:")
    print(classify_risk_from_transaction({
        "amount_inr": 42000, "hour_of_day": 2, "merchant_category": "Digital",
        "transaction_type": "UPI", "city": "Dubai", "day_of_week": "Sunday",
        "is_international": True, "velocity_flag": True, "geo_anomaly_flag": True,
        "high_amount_flag": True,
    }))
