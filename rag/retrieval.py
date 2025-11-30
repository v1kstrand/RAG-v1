from typing import Any, List, Sequence

import torch
from tqdm import tqdm

from .embeddings import BiEncoder


def build_gold_mask(gold_ids_per_query: List[List[int]], num_docs: int, device: str) -> torch.Tensor:
    mask = torch.zeros(len(gold_ids_per_query), num_docs, dtype=torch.bool, device=device)
    for i, gold_ids in enumerate(gold_ids_per_query):
        if gold_ids:
            mask[i, gold_ids] = True
    return mask


def evaluate_hit_at_k(
    questions: Sequence[str],
    gold_mask: torch.Tensor,
    embeddings: torch.Tensor,
    encoder: BiEncoder,
    k: int,
    batch_size: int,
) -> tuple[int, float]:
    """Batched Hit@K evaluation."""
    hits = 0
    total = len(questions)
    iterator = range(0, total, batch_size)
    iterator = tqdm(iterator, desc="Evaluating Hit@K", disable=total <= batch_size)

    for start in iterator:
        end = min(start + batch_size, total)
        q_embs = encoder.embed_queries(questions[start:end])
        if q_embs.ndim == 1:
            q_embs = q_embs.unsqueeze(0)
        sims = q_embs.to(embeddings.device) @ embeddings.T
        _, top_idx = sims.topk(k, dim=1)

        batch_gold_mask = gold_mask[start:end]
        retrieved_mask = torch.zeros_like(batch_gold_mask)
        retrieved_mask.scatter_(1, top_idx, True)
        hits += (batch_gold_mask & retrieved_mask).any(dim=1).sum().item()

    return hits, hits / total


def retrieve_top_k(
    question: str,
    embeddings: torch.Tensor,
    texts: Sequence[str],
    encoder: BiEncoder,
    k: int,
) -> List[dict[str, Any]]:
    q_emb = encoder.embed_queries(question)
    if q_emb.ndim > 1:
        q_emb = q_emb.squeeze(0)

    sims = embeddings @ q_emb
    top_vals, top_idx = torch.topk(sims, k=k)

    results = []
    for score, idx in zip(top_vals.tolist(), top_idx.tolist()):
        results.append(
            {
                "doc_id": idx,
                "score": float(score),
                "text": texts[idx],
            }
        )
    return results
