import os, re, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

# Convert topic string to a safe filename key
def _topic_to_cache_key(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", topic.lower())[:50]

# Run the full ingestion pipeline for a topic and return vector store + all papers
def build_pipeline(user_topic: str, force_rebuild: bool = False):
    from src.topic_expander import expand_topic
    from src.paper_search   import search_all_sources
    from src.pdf_parser     import ingest_papers
    from src.chunker        import chunk_all_papers
    from src.vector_store   import VectorStore

    cache_key       = _topic_to_cache_key(user_topic)
    vector_store    = VectorStore()
    raw_papers_path = f"data/cache/{cache_key}_raw.json"

    if not force_rebuild and vector_store.load(cache_key) and os.path.exists(raw_papers_path):
        print("✅ Loaded from cache")
        return vector_store, pd.read_json(raw_papers_path)

    print(f"\n🔍 Topic: {user_topic}")

    search_queries = expand_topic(user_topic)
    for query in search_queries:
        print(f"  • {query}")

    raw_papers_df = search_all_sources(search_queries)
    if raw_papers_df.empty:
        raise ValueError("No papers found. Try a different topic.")

    os.makedirs("data/cache", exist_ok=True)
    raw_papers_df.to_json(raw_papers_path, orient="records", indent=2)

    # Ingest and chunk ALL papers — no ranking gate
    ingested_papers = ingest_papers(raw_papers_df)
    all_chunks      = chunk_all_papers(ingested_papers)

    vector_store.build(all_chunks)
    vector_store.save(cache_key)
    print(f"\n✅ Pipeline complete — {len(all_chunks)} chunks from {len(raw_papers_df)} papers indexed")
    return vector_store, raw_papers_df


# Answer a user question using the built vector store
def ask_question(user_question: str, vector_store, run_evaluation: bool = False) -> dict:
    from src.reranker         import rerank_chunks
    from src.answer_generator import generate_answer
    from src.evaluator        import evaluate_rag_quality
    import config

    # Retrieve the best chunks from EVERY paper (diverse retrieval),
    # then rerank so the final context is still relevance-ordered.
    # This ensures follow-up questions are never limited to the subset of
    # papers that happened to surface in a previous answer.
    top_k = config.FINAL_RERANKED_COUNT
    top_per_paper = config.TOP_CHUNKS_PER_PAPER
    retrieved_chunks = vector_store.search_diverse(user_question, top_per_paper=top_per_paper)

    # Rerank for relevance
    top_chunks = rerank_chunks(user_question, retrieved_chunks, top_k=top_k)

    result = generate_answer(user_question, top_chunks)

    if run_evaluation:
        result["scores"] = evaluate_rag_quality(user_question, result["answer"], top_chunks)

    return result
