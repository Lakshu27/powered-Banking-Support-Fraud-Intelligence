"""
streamlit_app.py
------------------
Step 6: interactive dashboard for the AI-Powered Banking Support & Fraud
Intelligence System.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
from pipeline import run_pipeline

st.set_page_config(page_title="Banking Support & Fraud Intelligence", page_icon="🏦", layout="centered")

st.title("🏦 AI-Powered Banking Support & Fraud Intelligence")
st.caption("NLP intent detection + RAG-grounded responses + risk classification, in one flow.")

RISK_COLORS = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}

MERCHANT_CATEGORIES = ["E-commerce", "Food Delivery", "Digital", "Travel", "Groceries", "Utilities", "Entertainment", "Other"]
TRANSACTION_TYPES = ["UPI", "Debit Card", "Credit Card", "Net Banking", "ATM"]
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Kolkata", "Pune", "Dubai", "Other"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

with st.form("query_form"):
    query = st.text_area(
        "Customer query",
        placeholder="e.g. I see a transaction of ₹25,000 I didn't make",
        height=100,
    )

    with st.expander("Attach transaction details (optional — uses the trained fraud model instead of the text-only risk model)"):
        use_txn = st.checkbox("Score against a transaction record")
        c1, c2 = st.columns(2)
        amount = c1.number_input("Amount (INR)", min_value=0.0, value=1000.0, step=100.0)
        hour = c2.number_input("Hour of day (0-23)", min_value=0, max_value=23, value=14)
        c3, c4 = st.columns(2)
        merchant_category = c3.selectbox("Merchant category", MERCHANT_CATEGORIES)
        transaction_type = c4.selectbox("Transaction type", TRANSACTION_TYPES)
        c5, c6 = st.columns(2)
        city = c5.selectbox("City", CITIES)
        day_of_week = c6.selectbox("Day of week", DAYS)
        c7, c8 = st.columns(2)
        is_international = c7.checkbox("International transaction")
        velocity_flag = c8.checkbox("Unusual transaction frequency (velocity flag)")
        c9, c10 = st.columns(2)
        geo_anomaly_flag = c9.checkbox("Geographic anomaly")
        high_amount_flag = c10.checkbox("Above-normal amount")

    submitted = st.form_submit_button("Analyze")

if submitted and query.strip():
    transaction = None
    if use_txn:
        transaction = {
            "amount_inr": amount, "hour_of_day": hour,
            "merchant_category": merchant_category, "transaction_type": transaction_type,
            "city": city, "day_of_week": day_of_week,
            "is_international": is_international, "velocity_flag": velocity_flag,
            "geo_anomaly_flag": geo_anomaly_flag, "high_amount_flag": high_amount_flag,
        }

    with st.spinner("Analyzing query..."):
        result = run_pipeline(query, transaction=transaction)

    st.subheader("Result")

    col1, col2, col3 = st.columns(3)
    col1.metric("Intent", result["intent"])
    col2.metric("Sentiment", f"{result['sentiment']} ({result['sentiment_confidence']:.0%})")
    col3.metric("Risk", f"{RISK_COLORS.get(result['risk'], '')} {result['risk']}")

    st.markdown("**Suggested response**")
    st.info(result["response"])

    st.markdown("**Suggested action**")
    st.warning(result["action"])

    with st.expander("Retrieved context (what the response is grounded in)"):
        if result["retrieved"]:
            for r in result["retrieved"]:
                st.markdown(f"- `{r['type']}` **{r['source']}** (relevance {r['score']}): {r['text']}")
        else:
            st.write("No relevant context retrieved.")

    st.caption(f"Intent classifier confidence: {result['intent_confidence']:.0%} · risk source: {'transaction model' if transaction else 'ticket-text model'}")

elif submitted:
    st.warning("Please enter a query first.")

st.divider()
st.caption(
    "Trained on the real support_tickets.csv / transactions.csv / policy docs provided. "
    "Retrieval runs on TF-IDF similarity (see src/rag_pipeline.py for the note on upgrading "
    "to sentence-transformers + FAISS). Set GROQ_API_KEY (free, no card) or "
    "ANTHROPIC_API_KEY to enable LLM-generated "
    "responses instead of the extractive fallback."
)
