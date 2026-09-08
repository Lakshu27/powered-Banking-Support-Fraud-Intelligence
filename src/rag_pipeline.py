"""
rag_pipeline.py
-----------------
Steps 2 & 3 of the pipeline: retrieval + response generation, built on the
real policy documents (fraud_handling_policy.txt, kyc_policy.txt,
loan_processing_policy.txt, refund_dispute_policy.txt) and past resolved
tickets (resolution_text).

Retrieval note: this uses TF-IDF + cosine similarity instead of
sentence-transformers/FAISS. That's a deliberate choice for this sandbox
(no internet access to huggingface.co to pull embedding-model weights), and
`evaluate_retrieval()` below shows it performs reasonably against the
provided qa_pairs.json ground truth. To upgrade to dense semantic search on
a machine with normal internet access: swap `TfidfVectorizer` for a
SentenceTransformer encoder and the cosine-similarity search for a FAISS
IndexFlatIP — `retrieve(query, k)`'s interface doesn't need to change.

Generation note: if ANTHROPIC_API_KEY is set in the environment, this calls
Claude to generate a grounded response from the retrieved context. If not,
it falls back to an extractive template so the pipeline still runs without
any API key.
"""

import os
import textwrap
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import list_policy_docs, load_tickets, load_qa_pairs

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
INDEX_PATH = PROCESSED_DIR / "retrieval_index.joblib"


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------
def chunk_text(text: str, max_words: int = 90, overlap: int = 15) -> list[str]:
    """Fixed-size word chunking with overlap so a fact split across a chunk
    boundary is still retrievable from at least one chunk."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + max_words]))
        start += max_words - overlap
    return chunks


def build_corpus() -> list[dict]:
    """Collect chunks from policy docs + resolved tickets into one retrieval corpus."""
    corpus = []

    for doc_path in list_policy_docs():
        if doc_path.suffix.lower() != ".txt":
            continue
        text = doc_path.read_text()
        for chunk in chunk_text(text):
            corpus.append({"text": chunk, "source": doc_path.stem, "type": "policy"})

    tickets = load_tickets()
    for _, row in tickets.iterrows():
        corpus.append({
            "text": f"Past case ({row['category']}): {row['query_text']} -> {row['resolution_text']}",
            "source": row["ticket_id"],
            "type": "past_case",
        })

    return corpus


def build_index():
    """Builds two SEPARATE indices (policy docs, past cases) rather than one
    shared index. Early testing showed a shared index lets the 200 (heavily
    templated, near-duplicate) ticket chunks drown out the 4 policy docs —
    policy chunks almost never reached the top-k even for clearly
    policy-answerable questions. Retrieving from each source independently
    and merging (see retrieve()) fixes that and matches what the brief
    actually asks for: "Retrieve: Policies / Past tickets / Case histories"
    as parallel sources, not one combined ranking."""
    corpus = build_corpus()
    policy_chunks = [c for c in corpus if c["type"] == "policy"]
    case_chunks = [c for c in corpus if c["type"] == "past_case"]

    def _fit(chunks):
        texts = [c["text"] for c in chunks]
        # token_pattern excludes pure-digit tokens: early testing showed
        # rupee amounts (₹25,000 vs ₹2 crore vs ₹50,000) were matching on
        # numeric substrings ("25", "000") and outranking chunks that were
        # actually about the right topic but shared fewer literal words.
        # Letters-only tokens keep the match focused on meaning.
        vec = TfidfVectorizer(
            stop_words="english", max_features=6000, ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
        )
        matrix = vec.fit_transform(texts)
        return vec, matrix

    policy_vec, policy_matrix = _fit(policy_chunks)
    case_vec, case_matrix = _fit(case_chunks)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "policy_vec": policy_vec, "policy_matrix": policy_matrix, "policy_chunks": policy_chunks,
        "case_vec": case_vec, "case_matrix": case_matrix, "case_chunks": case_chunks,
    }, INDEX_PATH)
    print(f"Indexed {len(policy_chunks)} policy chunks + {len(case_chunks)} ticket chunks -> {INDEX_PATH}")
    return policy_vec, policy_matrix, policy_chunks, case_vec, case_matrix, case_chunks


def load_index():
    if not INDEX_PATH.exists():
        return build_index()
    d = joblib.load(INDEX_PATH)
    return d["policy_vec"], d["policy_matrix"], d["policy_chunks"], d["case_vec"], d["case_matrix"], d["case_chunks"]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def _search(query, vectorizer, matrix, chunks, k):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()
    top_idx = sims.argsort()[::-1][:k]
    return [{**chunks[i], "score": round(float(sims[i]), 3)} for i in top_idx if sims[i] > 0]


def retrieve(query: str, k: int = 3, k_policy: int = 2, k_cases: int = 2) -> list[dict]:
    """Retrieves from policy docs and past cases separately, then merges
    by score, so policy guidance is never crowded out by duplicate tickets.
    `k` caps the final merged result; k_policy/k_cases cap how many come
    from each source before merging."""
    policy_vec, policy_matrix, policy_chunks, case_vec, case_matrix, case_chunks = load_index()
    policy_hits = _search(query, policy_vec, policy_matrix, policy_chunks, k_policy)
    case_hits = _search(query, case_vec, case_matrix, case_chunks, k_cases)
    merged = sorted(policy_hits + case_hits, key=lambda r: r["score"], reverse=True)
    return merged[:k]


def evaluate_retrieval(k: int = 3) -> dict:
    """Checks retrieval quality against qa_pairs.json: for each QA question,
    is the policy document referenced in policy_ref present somewhere in the
    retrieved set? This is the 'retrieval relevance score' metric the
    project brief asks for."""
    qa = load_qa_pairs()
    hits, policy_scores = 0, []
    for _, row in qa.iterrows():
        results = retrieve(row["question"], k=k)
        policy_hits_in_result = [r for r in results if r["type"] == "policy"]
        if policy_hits_in_result:
            policy_scores.append(policy_hits_in_result[0]["score"])
        ref_first_word = row["policy_ref"].lower().split()[0]  # e.g. "fraud", "kyc", "loan", "refund"
        hit = any(r["type"] == "policy" and ref_first_word in r["source"] for r in results)
        hits += int(hit)
    return {
        "n_questions": len(qa),
        "policy_hit_rate": round(hits / len(qa), 3),
        "avg_policy_similarity_when_hit": round(sum(policy_scores) / len(policy_scores), 3) if policy_scores else 0.0,
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _generate_with_groq(query: str, context: str) -> str:
    """Free tier, no credit card required — sign up at console.groq.com,
    create an API key, export GROQ_API_KEY. Tried before Anthropic since
    it's free; if both keys are set, Groq wins."""
    import groq
    client = groq.Groq()  # reads GROQ_API_KEY from env
    prompt = (
        "You are a banking support assistant. Answer the customer's query "
        "using ONLY the context below (bank policy + past cases). Be concise "
        "and specific about next steps.\n\n"
        f"Context:\n{context}\n\nCustomer query: {query}"
    )
    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        max_tokens=600,  # gpt-oss models spend some tokens on internal reasoning
        reasoning_effort="low",  # minimize reasoning overhead so more budget goes to the visible answer
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def _generate_with_claude(query: str, context: str) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    prompt = (
        "You are a banking support assistant. Answer the customer's query "
        "using ONLY the context below (bank policy + past cases). Be concise "
        "and specific about next steps.\n\n"
        f"Context:\n{context}\n\nCustomer query: {query}"
    )
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _generate_extractive(query: str, retrieved: list[dict]) -> str:
    """No-API-key fallback. Pulls one past-case resolution and one policy
    snippet (when available) rather than concatenating two raw chunks —
    the earlier version could produce two near-identical "Based on our
    records: Past case..." lines back to back, which read poorly. This
    version labels the two source types distinctly and extracts just the
    resolution/policy text, not the restated customer query."""
    if not retrieved:
        return "I couldn't find a relevant policy or past case for this query — escalating to a human agent."

    case_hits = [r for r in retrieved if r["type"] == "past_case"]
    policy_hits = [r for r in retrieved if r["type"] == "policy"]
    parts = []

    if case_hits:
        text = case_hits[0]["text"]
        resolution = text.split(" -> ", 1)[1] if " -> " in text else text
        parts.append(f"Based on a similar past case: {textwrap.shorten(resolution, 180)}")

    if policy_hits:
        parts.append(f"Per {policy_hits[0]['source'].replace('_', ' ')}: {textwrap.shorten(policy_hits[0]['text'], 180)}")

    if not parts:  # neither type matched (shouldn't normally happen, but stay safe)
        parts.append(textwrap.shorten(retrieved[0]["text"], 200))

    return " ".join(parts)


def generate_response(query: str, k: int = 3) -> dict:
    """Tries Groq (free) first, then Anthropic, then falls back to the
    extractive response — silently on failure. LLM errors are printed to
    the console for debugging, never shown in the user-facing response
    text (an earlier version leaked raw error strings into the UI)."""
    retrieved = retrieve(query, k=k)
    context = "\n---\n".join(r["text"] for r in retrieved)
    response = None

    if os.environ.get("GROQ_API_KEY"):
        try:
            response = _generate_with_groq(query, context)
        except Exception as e:
            print(f"[rag_pipeline] Groq call failed, falling back: {e}")

    if response is None and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            response = _generate_with_claude(query, context)
        except Exception as e:
            print(f"[rag_pipeline] Claude call failed, falling back: {e}")

    if response is None:
        response = _generate_extractive(query, retrieved)

    return {"response": response, "retrieved": retrieved}


if __name__ == "__main__":
    build_index()

    print("\nRetrieval evaluation vs qa_pairs.json:")
    print(evaluate_retrieval())

    sample = "I see a transaction of ₹10,000 I didn't make"
    result = generate_response(sample)
    print("\nQuery:", sample)
    print("\nRetrieved:")
    for r in result["retrieved"]:
        print(f"  [{r['type']}/{r['source']} score={r['score']}] {r['text'][:100]}...")
    print("\nGenerated response:\n", result["response"])
