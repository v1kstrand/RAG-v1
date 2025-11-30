import argparse
from pathlib import Path

from rag.config import DatasetConfig, EncoderConfig
from rag.data import build_corpus, load_techqa
from rag.embeddings import BiEncoder, save_embeddings


def parse_args() -> argparse.Namespace:
    defaults = EncoderConfig()
    parser = argparse.ArgumentParser(description="Encode TechQA passages and persist embeddings.")
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=defaults.embedding_path,
        help="Where to save the passage embedding tensor.",
    )
    parser.add_argument(
        "--model",
        default=defaults.model_name,
        help="Encoder model name (HF hub).",
    )
    parser.add_argument(
        "--device",
        default=defaults.device,
        help="Device to use for encoding (cuda or cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults.passage_batch_size,
        help="Batch size for passage encoding.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=defaults.max_length,
        help="Max token length for the encoder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder_cfg = EncoderConfig(
        model_name=args.model,
        device=args.device,
        passage_batch_size=args.batch_size,
        max_length=args.max_length,
        embedding_path=args.embedding_path,
    )

    ds = load_techqa(DatasetConfig())
    texts, _ = build_corpus(ds)
    encoder = BiEncoder(encoder_cfg)

    print(f"Encoding {len(texts)} passages with {encoder_cfg.model_name} on {encoder_cfg.device}")
    embeddings = encoder.encode_corpus(texts, batch_size=encoder_cfg.passage_batch_size)
    save_embeddings(embeddings, encoder_cfg.embedding_path)
    print(f"Saved embeddings to {encoder_cfg.embedding_path}")


if __name__ == "__main__":
    main()

