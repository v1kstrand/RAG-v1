import time
from pathlib import Path
from typing import Any, Dict, List

import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from rag.config import EncoderConfig, LLMConfig, DatasetConfig
from rag.data import load_techqa, build_corpus
from rag.embeddings import BiEncoder, load_embeddings
from rag.llm import QwenClient
from rag.pipeline import answer_question


# ---------- Config ----------
DEFAULT_EMBEDDING_PATH = ROOT / "data/embeddings/techqa_bge.pt"
ENCODER_DEVICE = "cuda"
LLM_DEVICE = "cuda"

PAGE_TITLE = "TechQA RAG Demo"
PAGE_ICON = "📚"


# ---------- Cached loaders ----------

@st.cache_resource(show_spinner="Loading dataset, encoder, embeddings and LLM…")
def load_pipeline():
    """Load corpus, embeddings, encoder and LLM once per process."""
    # Load TechQA and build corpus of passage texts
    ds = load_techqa(DatasetConfig())
    texts, _ = build_corpus(ds)

    # Encoder + embeddings
    enc_cfg = EncoderConfig(
        embedding_path=DEFAULT_EMBEDDING_PATH,
        device=ENCODER_DEVICE,
    )
    encoder = BiEncoder(enc_cfg)
    embeddings = load_embeddings(enc_cfg.embedding_path, enc_cfg.device)

    # LLM
    llm_cfg = LLMConfig(device=LLM_DEVICE)
    llm = QwenClient(llm_cfg)

    return texts, embeddings, encoder, llm


# ---------- UI helpers ----------

def render_contexts(contexts: List[Dict[str, Any]]) -> None:
    if not contexts:
        st.info("No retrieved contexts to display.")
        return

    st.subheader("Retrieved contexts")

    for i, ctx in enumerate(contexts, start=1):
        # Try to be robust to different result shapes
        text = ctx.get("text") if isinstance(ctx, dict) else str(ctx)
        score = None
        doc_id = None

        if isinstance(ctx, dict):
            score = ctx.get("score")
            doc_id = ctx.get("doc_id") or ctx.get("id")

        with st.expander(f"Context #{i}  "
                         f"{f'(score={score:.3f})' if isinstance(score, (int, float)) else ''}"
                         f"{f'  [doc_id={doc_id}]' if doc_id is not None else ''}"):
            st.write(text)


# ---------- Main app ----------

def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

    st.title("TechQA RAG on NVIDIA A100")
    st.caption(
        "Retrieval-Augmented Generation over the TechQA-RAG-Eval dataset using "
        "`BAAI/bge-large-en-v1.5` embeddings + FAISS-GPU and a Qwen LLM."
    )

    # Check that embeddings exist
    if not DEFAULT_EMBEDDING_PATH.exists():
        st.error(
            f"Embedding file not found at `{DEFAULT_EMBEDDING_PATH}`.\n\n"
            "Run the embedding script first, for example:\n\n"
            "```bash\n"
            "python main.py embed --device cuda "
            "--embedding-path data/embeddings/techqa_bge.pt\n"
            "```"
        )
        st.stop()

    # Sidebar config / info
    with st.sidebar:
        st.header("Settings")
        st.markdown(
            f"""
- **Encoder device:** `{ENCODER_DEVICE}`
- **LLM device:** `{LLM_DEVICE}`
- **Embeddings path:** `{DEFAULT_EMBEDDING_PATH}`
            """
        )
        k = st.slider("Top-k passages", min_value=1, max_value=10, value=5, step=1)
        st.markdown("---")
        st.markdown(
            "This app wraps the existing `rag.pipeline.answer_question` "
            "function and keeps your models + embeddings cached in memory."
        )

    # Load pipeline objects (cached)
    try:
        texts, embeddings, encoder, llm = load_pipeline()
    except Exception as e:
        st.exception(e)
        st.stop()

    # Main layout
    col_left, col_right = st.columns([2.2, 1.8])

    with col_left:
        st.subheader("Ask a question")

        default_q = "How do I resolve NCCL connection issues?"
        query = st.text_area(
            "Question",
            value=default_q,
            height=100,
            help="Any TechQA-style support question over the underlying IBM docs.",
        )

        ask_button = st.button("Run RAG 🔎", type="primary")

        if ask_button and query.strip():
            with st.spinner("Retrieving passages and generating answer…"):
                t0 = time.perf_counter()
                result = answer_question(
                    query.strip(),
                    embeddings,
                    texts,
                    encoder,
                    llm,
                    k=k,
                )
                t1 = time.perf_counter()

            answer = result.get("answer", "")
            contexts = result.get("contexts") or result.get("hits") or []

            st.markdown("### Answer")
            if answer:
                st.success(answer)
            else:
                st.warning("No answer returned from the pipeline.")

            st.caption(f"End-to-end RAG latency: {(t1 - t0):.2f} seconds")

            # Show contexts below in the left column
            render_contexts(contexts)

    with col_right:
        st.subheader("How this works")
        st.markdown(
            """
**Pipeline overview**

1. Encode the question with `BAAI/bge-large-en-v1.5` on GPU.
2. Retrieve top-k passages with FAISS-GPU (cosine similarity).
3. Build a context-aware prompt from those passages.
4. Generate an answer with the Qwen LLM.

All heavy components (dataset, encoder, embeddings, LLM) are cached with
`st.cache_resource`, so they only load once per session.
            """
        )

        st.markdown("---")
        st.subheader("Tips")
        st.markdown(
            """
- If you change the embedding path or device, update the constants at the top of
  `app_streamlit.py`.
- If your `answer_question` result dict uses a different key than `contexts`
  (e.g. `hits`), adjust the `contexts = ...` line accordingly.
- To experiment with different LLMs, change the defaults in `LLMConfig`.
            """
        )


if __name__ == "__main__":
    main()