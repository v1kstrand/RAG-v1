import math
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .config import EncoderConfig


class BiEncoder:
    """Wrapper around BGE encoder for passage + query embeddings."""

    def __init__(self, cfg: EncoderConfig) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(cfg.model_name).to(self.device)
        self.model.eval()

    def _amp_ctx(self):
        return (
            torch.amp.autocast("cuda", dtype=self.cfg.amp_dtype)
            if self.device.startswith("cuda")
            else nullcontext()
        )

    def _embed_with_prefix(self, texts: str | Iterable[str], prefix: str) -> torch.Tensor:
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        prefixed = [f"{prefix}{t}" for t in items]

        with torch.no_grad():
            inputs = self.tokenizer(
                prefixed,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to(self.device)

            with self._amp_ctx():
                outputs = self.model(**inputs).last_hidden_state[:, 0, :]

        embeddings = F.normalize(outputs.detach(), p=2, dim=-1).to(torch.float32)
        return embeddings[0] if single else embeddings

    def embed_queries(self, texts: str | Iterable[str]) -> torch.Tensor:
        """Embed one or many queries with the BGE query prefix."""
        return self._embed_with_prefix(texts, "query: ")

    def embed_passages(self, texts: str | Iterable[str]) -> torch.Tensor:
        """Embed one or many passages with the BGE passage prefix."""
        return self._embed_with_prefix(texts, "passage: ")

    def encode_corpus(
        self, texts: Sequence[str], batch_size: int | None = None, show_progress: bool = True
    ) -> torch.Tensor:
        batch_size = batch_size or self.cfg.passage_batch_size
        num_docs = len(texts)
        hidden = self.model.config.hidden_size
        all_embs = torch.zeros((num_docs, hidden), device="cpu", dtype=torch.float32)
        num_batches = math.ceil(num_docs / batch_size)

        iterator = range(num_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding corpus")

        for b in iterator:
            start, end = b * batch_size, min((b + 1) * batch_size, num_docs)
            batch_embs = self.embed_passages(texts[start:end])
            all_embs[start:end] = batch_embs.cpu()

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return all_embs


def save_embeddings(embeddings: torch.Tensor, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    return output_path


def load_embeddings(path: Path, device: str) -> torch.Tensor:
    return torch.load(path, map_location=device)

