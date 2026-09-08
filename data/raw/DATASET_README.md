# Banking AI Support System — Dataset Documentation

## Overview
Synthetic dataset generated for a RAG-based banking customer support AI system.
All data is synthetic and does not contain real customer information.

## Files

### 01_support_tickets.csv (200 rows)
Customer support ticket history across 4 categories.

| Column | Description |
|--------|-------------|
| ticket_id | Unique ticket identifier (TKT-XXXXX) |
| date_created | Date ticket was raised |
| customer_id | Anonymized customer identifier |
| channel | Contact channel (Web Chat, Mobile App, Phone, Email, Branch) |
| category | Fraud/Unauthorized, Loan, KYC, Account Access |
| sub_category | More specific issue description |
| query_text | Full customer query text |
| sentiment | Customer emotional state |
| risk_level | High / Medium / Low |
| resolution_text | Agent resolution provided |
| resolution_time_minutes | Time to resolve |
| resolved_by | Agent name or Auto-Resolved |
| customer_satisfaction | Rating 1-5 |
| escalated | Whether ticket was escalated |

**Use for**: Intent classification training, response generation, sentiment analysis

### 02_transactions.csv (2,000 rows)
Transaction records with fraud labels for risk classification.

| Column | Description |
|--------|-------------|
| transaction_id | Unique transaction ID |
| account_id | Anonymized account |
| timestamp | Transaction date and time |
| amount_inr | Transaction amount in INR |
| merchant_name | Merchant name |
| merchant_category | Merchant business type |
| transaction_type | UPI / Debit Card / Credit Card / Net Banking / ATM |
| city | Transaction location |
| hour_of_day | Hour (0-23) |
| day_of_week | Day name |
| is_international | Yes/No |
| velocity_flag | Unusual transaction frequency detected |
| geo_anomaly_flag | Geographic anomaly detected |
| high_amount_flag | Amount above normal threshold |
| fraud_label | 1=Fraud, 0=Legitimate |
| fraud_reason | Reason (fraud cases only) |

**Fraud rate**: ~12% | **Use for**: Fraud classification ML model

### fraud_handling_policy.txt
Comprehensive fraud handling SOP including:
- Customer liability framework (RBI zero-liability)
- Risk classification (High/Medium/Low)
- SLA timelines
- Chargeback process
- Regulatory compliance requirements

### kyc_policy.txt
KYC & AML policy document including:
- Regulatory framework (PMLA, RBI)
- Document requirements
- V-CIP process
- Periodic review schedules
- Account restriction rules

### loan_processing_policy.txt
Retail loan policy including:
- Eligibility criteria and CIBIL requirements
- Document checklists by product
- Processing timelines
- Interest rate bands
- Foreclosure rules

### refund_dispute_policy.txt
Refund and dispute resolution policy including:
- Auto-reversal timelines (RBI mandated)
- Merchant dispute chargeback process
- Customer compensation rules
- Escalation matrix

### 04_qa_pairs.json (20 entries)
Structured Q&A pairs for RAG pipeline evaluation and fine-tuning.

| Field | Description |
|-------|-------------|
| id | QA pair identifier |
| category | Fraud / Loan / KYC / Account Access |
| question | Sample customer question |
| answer | Ideal agent response |
| policy_ref | Policy section reference |
| risk_level | High / Medium / Low |
| suggested_action | Recommended next action |

## Recommended Usage

```
Pipeline:
  tickets + qa_pairs  →  Intent Classifier (NLP)
  policy docs         →  Vector DB (FAISS/ChromaDB) for RAG
  transactions        →  Fraud Classification Model (XGBoost/Random Forest)
  qa_pairs            →  Evaluation / ground truth for RAG retrieval
```

## Notes
- All amounts in Indian Rupees (INR)
- All data is synthetic — no real PII
- Policy documents reflect common Indian banking regulations (RBI, PMLA, FEMA)
- Expand tickets dataset by augmenting with paraphrasing models
