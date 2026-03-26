import json
import config
from src.topic_expander import _call_llm

def rerank_chunks(user_question: str, retrieved_chunks: list, top_k: int = None) -> list:
    if not retrieved_chunks:
        return retrieved_chunks

    # Score each chunk for relevance — but NEVER drop chunks or destroy diversity on failure
    passages = "\n\n".join(
        f"[{i}] {c['paper_title']} ({c.get('year', '')}):\n{c['text'][:300]}"
        for i, c in enumerate(retrieved_chunks)
    )
    prompt = (
        f"Rate each passage's relevance to the question on a scale of 0 to 1.\n"
        f"Question: {user_question}\n\n{passages}\n\n"
        f"Return ONLY a JSON array like: "
        f'[{{"index": 0, "score": 0.9}}, {{"index": 1, "score": 0.4}}, ...]'
        f"\nEvery index from 0 to {len(retrieved_chunks) - 1} must appear exactly once."
    )

    try:
        text = _call_llm(prompt)
        # Strip markdown code fences if present
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        scored = json.loads(text)

        # Apply scores — only if we got valid entries back
        scores_applied = 0
        for item in scored:
            idx = item.get("index")
            if idx is not None and 0 <= idx < len(retrieved_chunks):
                retrieved_chunks[idx]["rerank_score"] = float(item.get("score", 0))
                scores_applied += 1

        # Only re-sort if we successfully scored at least half the chunks
        # This prevents partial failures from destroying diversity
        if scores_applied >= len(retrieved_chunks) // 2:
            retrieved_chunks = sorted(
                retrieved_chunks,
                key=lambda c: c.get("rerank_score", c.get("retrieval_score", 0)),
                reverse=True
            )

    except Exception as e:
        print(f"  [reranker] fallback to retrieval order: {e}")
        # On any failure, keep the original diversity-sorted order untouched

    # Apply top_k only if explicitly requested
    if top_k is not None and len(retrieved_chunks) > top_k:
        return retrieved_chunks[:top_k]
    return retrieved_chunks
