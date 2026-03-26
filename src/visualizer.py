import numpy as np

def build_umap_figure(vector_store):
    """Generate a Plotly UMAP scatter of all chunk embeddings, coloured by paper."""
    try:
        import umap
        import plotly.express as px
        import pandas as pd

        chunks  = vector_store.stored_chunks
        index   = vector_store.faiss_index
        n       = index.ntotal

        # Reconstruct all vectors from the FAISS flat index
        vectors = np.zeros((n, index.d), dtype="float32")
        for i in range(n):
            vectors[i] = index.reconstruct(i)

        # Reduce to 2D
        reducer   = umap.UMAP(n_components=2, random_state=42, min_dist=0.3, n_neighbors=15)
        embedding = reducer.fit_transform(vectors)

        df = pd.DataFrame({
            "x":       embedding[:, 0],
            "y":       embedding[:, 1],
            "paper":   [c["paper_title"][:55] + "…" if len(c["paper_title"]) > 55 else c["paper_title"] for c in chunks],
            "section": [c.get("section", "unknown").title() for c in chunks],
            "method":  [c.get("chunk_method", "fixed") for c in chunks],
            "preview": [c["text"][:120] + "…" for c in chunks],
        })

        fig = px.scatter(
            df, x="x", y="y",
            color="paper",
            symbol="section",
            hover_data={"x": False, "y": False, "preview": True, "section": True, "method": True},
            title="Embedding Space — each dot is a chunk, colour = paper, shape = section",
            labels={"x": "UMAP-1", "y": "UMAP-2"},
            template="plotly_white",
            height=540,
        )
        fig.update_traces(marker=dict(size=7, opacity=0.82))
        fig.update_layout(
            font_family="Georgia, serif",
            title_font_size=14,
            legend=dict(orientation="v", x=1.02, y=1, font_size=10),
            paper_bgcolor="#faf6ef",
            plot_bgcolor="#faf6ef",
        )
        return fig

    except Exception as e:
        print(f"  [visualizer] UMAP failed: {e}")
        return None
