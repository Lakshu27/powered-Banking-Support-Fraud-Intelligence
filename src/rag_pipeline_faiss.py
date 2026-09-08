"""
rag_pipeline_faiss.py
-----------------------
Dense-embedding version of the RAG retrieval layer: sentence-transformers
for embeddings, real FAISS for similarity search. Same interface as
rag_pipeline.py (retrieve, generate_response, evaluate_retrieval,
build_index) — this is a drop-in swap, not a rewrite of the pipeline.

Why this exists as a SEPARATE file from rag_pipeline.py: the TF-IDF version
was built in a sandboxed dev environment with no internet access to
huggingface.co, so the embedding model can't download there. Your machine
has internet, so this version will work directly for you. Once you've
confirmed it works, you can either keep both (and choose which to import in
pipeline.py) or delete rag_pipeline.py and rename this file.

First run will download the embedding model (~90MB, one-time, needs
internet) — that's expected and only happens once; it's cached locally
after that.

Usage: identical to rag_pipeline.py
    python src/rag_pipeline_faiss.py
"""

import os
import textwrap
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from data_loader import list_policy_docs, load_tickets, load_qa_pairs

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
POLICY_INDEX_PATH = PROCESSED_DIR / "faiss_policy.index"
CASE_INDEX_PATH = PROCESSED_DIR / "faiss_case.index"
POLICY_META_PATH = PROCESSED_DIR / "faiss_policy_meta.npy"
CASE_META_PATH = PROCESSED_DIR / "faiss_case_meta.npy"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, good enough for this corpus size
_model = None  # lazy-loaded singleton so repeated calls don't reload the model


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


# ---------------------------------------------------------------------------
# Index building (same chunking + dual-index design as rag_pipeline.py —
# keeping policies and past cases separate is what fixed the 15% -> 75%
# policy-hit-rate issue found during development; that fix applies here too)
# ---------------------------------------------------------------------------
def chunk_text(text: str, max_words: int = 90, overlap: int = 15) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + max_words]))
        start += max_words - overlap
    return chunks


def _build_corpus():
    policy_chunks, case_chunks = [], []

    for doc_path in list_policy_docs():
        if doc_path.suffix.lower() != ".txt":
            continue
        for chunk in chunk_text(doc_path.read_text()):
            policy_chunks.append({"text": chunk, "source": doc_path.stem, "type": "policy"})

    for _, row in load_tickets().iterrows():
        case_chunks.append({
            "text": f"Past case ({row['category']}): {row['query_text']} -> {row['resolution_text']}",
            "source": row["ticket_id"],
            "type": "past_case",
        })

    return policy_chunks, case_chunks


def build_index():
    policy_chunks, case_chunks = _build_corpus()
    model = get_model()

    def _build_faiss(chunks):
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype="float32")
        index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product on normalized vectors = cosine similarity
        index.add(embeddings)
        return index

    policy_index = _build_faiss(policy_chunks)
    case_index = _build_faiss(case_chunks)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(policy_index, str(POLICY_INDEX_PATH))
    faiss.write_index(case_index, str(CASE_INDEX_PATH))
    np.save(POLICY_META_PATH, np.array(policy_chunks, dtype=object))
    np.save(CASE_META_PATH, np.array(case_chunks, dtype=object))

    print(f"Indexed {len(policy_chunks)} policy chunks + {len(case_chunks)} ticket chunks (FAISS, {EMBEDDING_MODEL})")
    return policy_index, policy_chunks, case_index, case_chunks


def load_index():
    if not (POLICY_INDEX_PATH.exists() and CASE_INDEX_PATH.exists()):
        return build_index()
    policy_index = faiss.read_index(str(POLICY_INDEX_PATH))
    case_index = faiss.read_index(str(CASE_INDEX_PATH))
    policy_chunks = np.load(POLICY_META_PATH, allow_pickle=True).tolist()
    case_chunks = np.load(CASE_META_PATH, allow_pickle=True).tolist()
    return policy_index, policy_chunks, case_index, case_chunks


# ---------------------------------------------------------------------------
# Retrieval — same merged dual-source design as rag_pipeline.py
# ---------------------------------------------------------------------------
def _search(query_vec, index, chunks, k):
    if index.ntotal == 0:
        return []
    scores, idx = index.search(query_vec, min(k, index.ntotal))
    results = []
    for score, i in zip(scores[0], idx[0]):
        if i == -1:
            continue
        results.append({**chunks[i], "score": round(float(score), 3)})
    return results


def retrieve(query: str, k: int = 3, k_policy: int = 2, k_cases: int = 2) -> list[dict]:
    policy_index, policy_chunks, case_index, case_chunks = load_index()
    model = get_model()
    q_vec = np.asarray(model.encode([query], normalize_embeddings=True), dtype="float32")

    policy_hits = _search(q_vec, policy_index, policy_chunks, k_policy)
    case_hits = _search(q_vec, case_index, case_chunks, k_cases)
    merged = sorted(policy_hits + case_hits, key=lambda r: r["score"], reverse=True)
    return merged[:k]


def evaluate_retrieval(k: int = 3) -> dict:
    qa = load_qa_pairs()
    hits, policy_scores = 0, []
    for _, row in qa.iterrows():
        results = retrieve(row["question"], k=k)
        policy_hits_in_result = [r for r in results if r["type"] == "policy"]
        if policy_hits_in_result:
            policy_scores.append(policy_hits_in_result[0]["score"])
        ref_first_word = row["policy_ref"].lower().split()[0]
        hit = any(r["type"] == "policy" and ref_first_word in r["source"] for r in results)
        hits += int(hit)
    return {
        "n_questions": len(qa),
        "policy_hit_rate": round(hits / len(qa), 3),
        "avg_policy_similarity_when_hit": round(sum(policy_scores) / len(policy_scores), 3) if policy_scores else 0.0,
    }


# ---------------------------------------------------------------------------
# Generation — same as rag_pipeline.py, with Groq (free) tried first
# ---------------------------------------------------------------------------
def _generate_with_groq(query: str, context: str) -> str:
    """Free tier, no credit card required — sign up at console.groq.com,
    create an API key, export GROQ_API_KEY. Tried before Anthropic since
    it's free; if both keys are set, Groq wins."""
    import groq
    client = groq.Groq()
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
    client = anthropic.Anthropic()
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
            print(f"[rag_pipeline_faiss] Groq call failed, falling back: {e}")

    if response is None and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            response = _generate_with_claude(query, context)
        except Exception as e:
            print(f"[rag_pipeline_faiss] Claude call failed, falling back: {e}")

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
    for r in result["retrieved"]:
        print(f"  [{r['type']}/{r['source']} score={r['score']}] {r['text'][:100]}...")
    print("\nGenerated response:\n", result["response"])
