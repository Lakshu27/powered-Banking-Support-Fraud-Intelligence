"""
pipeline.py
------------
Orchestrates the full flow for a single customer query:
  query -> intent + sentiment -> retrieve context -> generate response
        -> classify risk -> suggest action

Called by the Streamlit app (app/streamlit_app.py) and usable directly.
"""

from intent_classifier import predict_intent
from rag_pipeline import generate_response
from risk_classifier import classify_risk_from_text, classify_risk_from_transaction
from action_layer import suggest_action


def run_pipeline(query: str, transaction: dict = None) -> dict:
    intent_result = predict_intent(query)
    intent, sentiment = intent_result["intent"], intent_result["sentiment"]

    gen = generate_response(query)

    if transaction:
        risk = classify_risk_from_transaction(transaction)
    else:
        risk = classify_risk_from_text(query, category=intent)

    action = suggest_action(intent, risk)

    return {
        "query": query,
        "intent": intent,
        "intent_confidence": intent_result["confidence"],
        "sentiment": sentiment,
        "sentiment_confidence": intent_result["sentiment_confidence"],
        "retrieved": gen["retrieved"],
        "response": gen["response"],
        "risk": risk,
        "action": action,
    }


if __name__ == "__main__":
    result = run_pipeline("I see a transaction of ₹25,000 I didn't make")
    for k, v in result.items():
        if k != "retrieved":
            print(f"{k}: {v}")
    print("retrieved sources:", [(r["type"], r["source"]) for r in result["retrieved"]])
