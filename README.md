# TechQA RAG (modular)

This repository refactors the original notebook-style `main.py` into a small, modular RAG toolkit for the TechQA dataset. Core functionality lives in the `rag/` package, while runnable entrypoints are in `scripts/` and the `main.py` CLI.

## Layout
- `rag/`: shared modules (config, data prep, embeddings, retrieval, LLM prompt/answer pipeline).
- `scripts/`: task-specific entrypoints for embedding, evaluation, and question answering.
- `main.py`: convenient CLI with the same subcommands as the scripts.
- `main.ipynb`: the original notebook (kept for reference).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Notes:
- Models are large; a GPU is strongly recommended for the encoder (BGE) and the Qwen LLM.
- If you want a smaller footprint, point `--encoder-model` and `--llm-model` to lighter HF models.

## Quickstart (CLI)
All functionality is available from `main.py` via subcommands.

```bash
# 1) Encode TechQA passages
python main.py embed --device cuda --embedding-path data/embeddings/techqa_bge.pt

# 2) Evaluate Hit@K retrieval
python main.py eval --k 5 --batch-size 32 --device cuda

# 3) Ask a question with RAG
python main.py ask --question "How do I resolve NCCL connection issues?" \
    --k 5 --encoder-device cuda --llm-device cuda
```

## Script-by-script examples
### Embed the TechQA corpus
```bash
python scripts/embed_corpus.py \
  --device cuda \
  --model BAAI/bge-large-en-v1.5 \
  --batch-size 256 \
  --max-length 512 \
  --embedding-path data/embeddings/techqa_bge.pt
```
- Downloads TechQA via `datasets`.
- Encodes unique contexts with the BGE encoder.
- Saves a `torch.Tensor` to `data/embeddings/techqa_bge.pt` by default.

### Evaluate retrieval Hit@K
```bash
python scripts/eval_retrieval.py \
  --embedding-path data/embeddings/techqa_bge.pt \
  --k 5 \
  --batch-size 32 \
  --device cuda \
  --split train
```
- Loads saved passage embeddings.
- Embeds questions in batches and reports Hit@K.

### Ask a question with RAG
```bash
python scripts/ask.py \
  --question "How do I resolve NCCL connection issues?" \
  --k 5 \
  --embedding-path data/embeddings/techqa_bge.pt \
  --encoder-device cuda \
  --llm-device cuda \
  --llm-model Qwen/Qwen2.5-7B-Instruct
```
- Retrieves top-k passages with the bi-encoder.
- Builds a context-aware prompt and generates an answer with Qwen.
- Prints the answer plus doc ids and scores for traceability.

## Using the Python API
```python
from pathlib import Path
from rag.config import EncoderConfig, LLMConfig, DatasetConfig
from rag.data import build_corpus, load_techqa
from rag.embeddings import BiEncoder, load_embeddings
from rag.llm import QwenClient
from rag.pipeline import answer_question

ds = load_techqa(DatasetConfig())
texts, _ = build_corpus(ds)

enc_cfg = EncoderConfig(embedding_path=Path("data/embeddings/techqa_bge.pt"), device="cuda")
llm_cfg = LLMConfig(device="cuda")

encoder = BiEncoder(enc_cfg)
llm = QwenClient(llm_cfg)
embeddings = load_embeddings(enc_cfg.embedding_path, enc_cfg.device)

result = answer_question("How do I resolve NCCL connection issues?", embeddings, texts, encoder, llm, k=5)
print(result["answer"])
```

## Default paths and caching
- Embeddings default to `data/embeddings/techqa_bge.pt` (configurable via CLI flags).
- The `datasets` library caches TechQA under `~/.cache/huggingface/`.
- `.gitignore` already ignores `data/embeddings/` and `*.pt` to avoid committing large artifacts.
