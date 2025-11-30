import argparse
from pathlib import Path

from rag.config import DatasetConfig, EncoderConfig
from rag.data import build_corpus, build_questions_and_gold, load_techqa
from rag.embeddings import BiEncoder, load_embeddings
from rag.retrieval import build_gold_mask, evaluate_hit_at_k


def parse_args() -> argparse.Namespace:
    defaults = EncoderConfig()
    parser = argparse.ArgumentParser(description="Evaluate batched Hit@K retrieval on TechQA.")
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=defaults.embedding_path,
        help="Path to passage embeddings created by embed_corpus.py.",
    )
    parser.add_argument("--model", default=defaults.model_name, help="Encoder model for queries.")
    parser.add_argument("--device", default=defaults.device, help="Device for encoding (cuda or cpu).")
    parser.add_argument("--k", type=int, default=defaults.k, help="K for Hit@K.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults.query_batch_size,
        help="Query batch size.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to evaluate (default: train).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder_cfg = EncoderConfig(
        model_name=args.model,
        device=args.device,
        query_batch_size=args.batch_size,
        k=args.k,
        embedding_path=args.embedding_path,
    )

    if not encoder_cfg.embedding_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {encoder_cfg.embedding_path}. Run scripts/embed_corpus.py first."
        )

    ds = load_techqa(DatasetConfig())
    texts, fname_to_idx = build_corpus(ds)
    questions, gold_ids = build_questions_and_gold(ds, fname_to_idx, split=args.split)
    print(f"Evaluating retrieval on {len(questions)} {args.split} questions")

    encoder = BiEncoder(encoder_cfg)
    embeddings = load_embeddings(encoder_cfg.embedding_path, encoder_cfg.device).to(encoder_cfg.device)
    gold_mask = build_gold_mask(gold_ids, embeddings.shape[0], encoder_cfg.device)

    hits, hit_at_k = evaluate_hit_at_k(
        questions,
        gold_mask,
        embeddings,
        encoder,
        k=encoder_cfg.k,
        batch_size=encoder_cfg.query_batch_size,
    )
    print(f"Hit@{encoder_cfg.k}: {hits}/{len(questions)} = {hit_at_k:.3f}")


if __name__ == "__main__":
    main()

