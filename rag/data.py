from typing import Dict, Iterable, List, Tuple
from datasets import DatasetDict, load_dataset

from .config import DatasetConfig


def load_techqa(dataset_cfg: DatasetConfig) -> DatasetDict:
    """Load the TechQA dataset (cached locally by `datasets`)."""
    return load_dataset(dataset_cfg.name)


def assert_unique_contexts(ds: DatasetDict, splits: Iterable[str] | None = None) -> None:
    filename_to_text: Dict[str, str] = {}
    text_to_filename: Dict[str, str] = {}
    splits = splits or list(ds.keys())

    for split in splits:
        for row in ds[split]:
            for ctx in row["contexts"]:
                fname = ctx["filename"]
                text = ctx["text"].strip()
                if fname in filename_to_text and filename_to_text[fname] != text:
                    raise ValueError(f"Duplicate filename with different text: {fname}")
                if text in text_to_filename and text_to_filename[text] != fname:
                    raise ValueError("Duplicate text with different filename detected")
                filename_to_text[fname] = text
                text_to_filename[text] = fname


def build_corpus(ds: DatasetDict, splits: Iterable[str] | None = None) -> Tuple[List[str], Dict[str, int]]:
    """Return unique context texts; index order defines doc_id for embeddings."""
    assert_unique_contexts(ds, splits)
    splits = splits or list(ds.keys())
    texts: List[str] = []
    fname_to_idx: Dict[str, int] = {}
    seen_filenames: set[str] = set()
    for split in splits:
        for row in ds[split]:
            for ctx in row["contexts"]:
                fname = ctx["filename"]
                if fname in seen_filenames:
                    continue
                seen_filenames.add(fname)
                fname_to_idx[fname] = len(texts)
                texts.append(ctx["text"].strip())
    return texts, fname_to_idx


def build_questions_and_gold(
    ds: DatasetDict, fname_to_idx: Dict[str, int], split: str = "train"
) -> Tuple[List[str], List[List[int]]]:
    """Extract questions and gold document ids for evaluation."""
    questions: List[str] = []
    gold_ids_per_query: List[List[int]] = []
    for row in ds[split]:
        question = row["question"]
        gold_doc_ids: List[int] = []
        for ctx in row["contexts"]:
            fname = ctx["filename"].strip()
            doc_id = fname_to_idx.get(fname)
            if doc_id is not None:
                gold_doc_ids.append(doc_id)
        questions.append(question)
        gold_ids_per_query.append(gold_doc_ids)
    return questions, gold_ids_per_query
