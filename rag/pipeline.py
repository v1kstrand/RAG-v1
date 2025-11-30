from typing import Any, Dict, Sequence

import torch

from .config import EncoderConfig, LLMConfig
from .embeddings import BiEncoder
from .llm import QwenClient
from .retrieval import retrieve_top_k


def ensure_embeddings_device(embeddings: torch.Tensor, device: str) -> torch.Tensor:
    if embeddings.device.type == device:
        return embeddings
    return embeddings.to(device)


def make_default_encoder(cfg: EncoderConfig | None = None) -> BiEncoder:
    cfg = cfg or EncoderConfig()
    return BiEncoder(cfg)


def make_default_llm(cfg: LLMConfig | None = None) -> QwenClient:
    cfg = cfg or LLMConfig()
    return QwenClient(cfg)


def answer_question(
    question: str,
    embeddings: torch.Tensor,
    texts: Sequence[str],
    encoder: BiEncoder,
    llm: QwenClient,
    k: int,
) -> Dict[str, Any]:
    embeddings = ensure_embeddings_device(embeddings, encoder.device)
    retrieved = retrieve_top_k(question, embeddings, texts, encoder, k)
    messages = llm.build_messages(question, retrieved)
    answer = llm.generate(messages)
    return {"answer": answer, "contexts": retrieved}
