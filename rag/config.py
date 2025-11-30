from dataclasses import dataclass
from pathlib import Path
import torch


DEFAULT_DATASET = "nvidia/TechQA-RAG-Eval"
DEFAULT_EMB_PATH = Path("data/embeddings/techqa_bge.pt")


def default_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EncoderConfig:
    model_name: str = "BAAI/bge-large-en-v1.5"
    max_length: int = 512
    passage_batch_size: int = 256
    query_batch_size: int = 32
    k: int = 5
    amp_dtype: torch.dtype = torch.float32
    device: str = default_device()
    embedding_path: Path = DEFAULT_EMB_PATH


@dataclass
class DatasetConfig:
    name: str = DEFAULT_DATASET


@dataclass
class LLMConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    device: str = default_device()

