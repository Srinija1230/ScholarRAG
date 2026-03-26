def evaluate_rag_quality(question: str, answer: str, chunks: list) -> dict:
    """LLM-based RAG evaluation scoring faithfulness, answer relevance, and context precision.

    Each metric is scored independently via the active LLM on a 0-1 scale:
    - Faithfulness:       are all claims in the answer supported by the context?
    - Answer Relevance:   does the answer actually address the question?
    - Context Precision:  are the retrieved chunks genuinely useful for the question?
    """
    try:
        from src.topic_expander import _call_llm
        import json, re

        context_text = "\n\n".join(f"[{i+1}] {c['text'][:400]}" for i, c in enumerate(chunks[:8]))

        prompt = f"""You are evaluating the quality of a RAG (retrieval-augmented generation) response.
Score each metric from 0.0 to 1.0 based on the definitions below.

QUESTION: {question}

RETRIEVED CONTEXT:
{context_text}

ANSWER: {answer[:600]}

Scoring definitions:
- faithfulness (0-1): What fraction of the answer's claims are directly supported by the context? 1.0 = fully grounded, 0.0 = entirely fabricated.
- answer_relevance (0-1): How well does the answer address the question? 1.0 = directly and completely answers it, 0.0 = off-topic.
- context_precision (0-1): What fraction of the retrieved chunks are actually useful for answering this question? 1.0 = all chunks relevant, 0.0 = none relevant.

Carefully evaluate the actual content above and assign honest scores. Then return ONLY valid JSON with your real scores, no explanation or markdown. Example format (use your own values, not these):
{{"faithfulness": 0.85, "answer_relevance": 0.92, "context_precision": 0.75}}"""

        raw = _call_llm(prompt)
        raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()
        scores = json.loads(raw)

        return {
            "faithfulness":      round(min(1.0, max(0.0, float(scores.get("faithfulness", 0)))), 3),
            "answer_relevance":  round(min(1.0, max(0.0, float(scores.get("answer_relevance", 0)))), 3),
            "context_precision": round(min(1.0, max(0.0, float(scores.get("context_precision", 0)))), 3),
        }
    except Exception as e:
        print(f"  [Evaluator] scoring failed: {e}")
        return {"faithfulness": 0.0, "answer_relevance": 0.0, "context_precision": 0.0}