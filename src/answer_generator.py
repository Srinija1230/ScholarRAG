import config

ANSWER_PROMPT = """You are a research assistant. Read all the context passages below, then write ONE single cohesive answer to the question.

Rules:
- Write a single flowing response — do NOT structure it as "Paper 1 says... Paper 2 says..."
- Blend insights from all relevant passages together into unified paragraphs
- Cite claims inline like [Smith et al., 2022] naturally within the sentences
- Do not repeat the question, do not use headings, do not give a per-paper breakdown
- If the context does not contain the answer, say so in one sentence

Context:
{context}

Question: {question}

Answer:"""

OVERVIEW_PROMPT = """You are a research assistant. Based on the research passages below, write a single cohesive overview of the topic: "{topic}".

Rules:
- Write 3 to 5 paragraphs as one flowing response
- Cover what the topic is, what approaches exist, key findings, and open challenges
- Blend insights from all passages naturally — do NOT summarize passage by passage
- Cite claims inline like [Smith et al., 2022] where relevant
- Do not use headings or bullet points

Context:
{context}

Overview:"""


def _call_llm(prompt: str) -> str:
    provider  = config.ACTIVE_PROVIDER
    model_cfg = config.MODEL_OPTIONS[provider]

    if provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=model_cfg["api_key"])
        return genai.GenerativeModel(model_cfg["chat_model"]).generate_content(prompt).text.strip()

    elif provider == "claude":
        import anthropic
        client = anthropic.Anthropic(api_key=model_cfg["api_key"])
        msg = client.messages.create(
            model=model_cfg["chat_model"], max_tokens=1024,
            messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip()

    elif provider == "llama":
        from groq import Groq
        client = Groq(api_key=model_cfg["api_key"])
        resp = client.chat.completions.create(
            model=model_cfg["chat_model"], max_tokens=1024,
            messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content.strip()

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=model_cfg["api_key"])
        resp = client.chat.completions.create(
            model=model_cfg["chat_model"], max_tokens=1024,
            messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content.strip()

    return "Unknown provider."


def _chunks_to_context(chunks: list) -> str:
    labelled = []
    for chunk in chunks:
        authors   = chunk.get("authors", "")
        last_name = authors.split(",")[0].split()[-1] if authors else "?"
        cite      = f"{last_name} et al., {chunk.get('year', 'n.d.')}"
        labelled.append(f"[{cite}] {chunk['text']}")
    return " ".join(labelled)


def generate_answer(user_question: str, top_chunks: list) -> dict:
    from collections import OrderedDict

    context_text = _chunks_to_context(top_chunks)

    try:
        answer_text = _call_llm(ANSWER_PROMPT.format(context=context_text, question=user_question))
    except Exception as e:
        answer_text = f"Error generating answer: {e}"

    # Sources list — one entry per unique paper
    seen = OrderedDict()
    for chunk in top_chunks:
        title = chunk["paper_title"]
        if title not in seen:
            seen[title] = {
                "title":   title,
                "authors": chunk.get("authors", ""),
                "year":    chunk.get("year", ""),
                "url":     chunk.get("paper_url", ""),
                "excerpt": chunk["text"][:200] + "…",
            }

    return {"answer": answer_text, "sources": list(seen.values())}


def generate_topic_overview(topic: str, vector_store) -> str:
    """Generate overview from top K chunks most relevant to the topic — pure RAG.
    Uses diverse retrieval so every fetched paper contributes to the overview,
    not just the handful that score highest on the raw embedding search."""
    import config
    from src.reranker import rerank_chunks

    top_k = config.FINAL_RERANKED_COUNT
    top_per_paper = config.TOP_CHUNKS_PER_PAPER
    # Pull the best chunks from all papers, then rerank to final top_k
    retrieved_chunks = vector_store.search_diverse(topic, top_per_paper=top_per_paper)
    top_chunks       = rerank_chunks(topic, retrieved_chunks, top_k=top_k)

    context_text = _chunks_to_context(top_chunks)

    try:
        return _call_llm(OVERVIEW_PROMPT.format(topic=topic, context=context_text))
    except Exception as e:
        return f"Could not generate overview: {e}"


def generate_topic_overview_with_scores(topic: str, vector_store, run_evaluation: bool = False) -> dict:
    """Like generate_topic_overview but returns a dict with overview text + optional scores."""
    import config
    from src.reranker import rerank_chunks

    top_k = config.FINAL_RERANKED_COUNT
    top_per_paper = config.TOP_CHUNKS_PER_PAPER
    retrieved_chunks = vector_store.search_diverse(topic, top_per_paper=top_per_paper)
    top_chunks       = rerank_chunks(topic, retrieved_chunks, top_k=top_k)

    context_text = _chunks_to_context(top_chunks)

    try:
        overview_text = _call_llm(OVERVIEW_PROMPT.format(topic=topic, context=context_text))
    except Exception as e:
        overview_text = f"Could not generate overview: {e}"

    scores = None
    if run_evaluation:
        try:
            from src.evaluator import evaluate_rag_quality
            scores = evaluate_rag_quality(topic, overview_text, top_chunks)
        except Exception as e:
            print(f"  [Evaluator] overview scoring failed: {e}")

    return {"overview": overview_text, "scores": scores}