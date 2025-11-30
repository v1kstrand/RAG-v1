import argparse
from pathlib import Path

from rag.config import DatasetConfig, EncoderConfig, LLMConfig
from rag.data import build_corpus, load_techqa
from rag.embeddings import BiEncoder, load_embeddings
from rag.llm import QwenClient
from rag.pipeline import answer_question, ensure_embeddings_device


def parse_args() -> argparse.Namespace:
    enc_defaults = EncoderConfig()
    llm_defaults = LLMConfig()
    parser = argparse.ArgumentParser(description="Ask a question with the RAG pipeline.")
    parser.add_argument("--question", "-q", required=True, help="Natural language question to answer.")
    parser.add_argument("--k", type=int, default=enc_defaults.k, help="Top-k documents to retrieve.")
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=enc_defaults.embedding_path,
        help="Path to passage embeddings created by embed_corpus.py.",
    )
    parser.add_argument("--encoder-model", default=enc_defaults.model_name, help="HF encoder model id.")
    parser.add_argument("--encoder-device", default=enc_defaults.device, help="Device for encoder (cuda or cpu).")
    parser.add_argument("--max-length", type=int, default=enc_defaults.max_length, help="Encoder max token length.")
    parser.add_argument("--llm-model", default=llm_defaults.model_name, help="LLM model id.")
    parser.add_argument("--llm-device", default=llm_defaults.device, help="Device for LLM (cuda or cpu).")
    parser.add_argument("--max-new-tokens", type=int, default=llm_defaults.max_new_tokens, help="LLM max tokens.")
    parser.add_argument("--temperature", type=float, default=llm_defaults.temperature, help="LLM temperature.")
    parser.add_argument("--top-p", type=float, default=llm_defaults.top_p, help="LLM top-p sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder_cfg = EncoderConfig(
        model_name=args.encoder_model,
        device=args.encoder_device,
        max_length=args.max_length,
        embedding_path=args.embedding_path,
        k=args.k,
    )
    llm_cfg = LLMConfig(
        model_name=args.llm_model,
        device=args.llm_device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if not encoder_cfg.embedding_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {encoder_cfg.embedding_path}. Run scripts/embed_corpus.py first."
        )

    ds = load_techqa(DatasetConfig())
    texts, _ = build_corpus(ds)
    encoder = BiEncoder(encoder_cfg)
    llm = QwenClient(llm_cfg)

    embeddings = load_embeddings(encoder_cfg.embedding_path, encoder_cfg.device)
    embeddings = ensure_embeddings_device(embeddings, encoder_cfg.device)

    result = answer_question(
        args.question,
        embeddings,
        texts,
        encoder,
        llm,
        k=encoder_cfg.k,
    )

    print("\nANSWER:\n")
    print(result["answer"])
    print("\nDOCS USED:")
    for ctx in result["contexts"]:
        print(f"- doc_id={ctx['doc_id']}, score={ctx['score']:.3f}")


if __name__ == "__main__":
    main()

