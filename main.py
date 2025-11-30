import argparse
from pathlib import Path

from rag.config import DatasetConfig, EncoderConfig, LLMConfig
from rag.data import build_corpus, build_questions_and_gold, load_techqa
from rag.embeddings import BiEncoder, load_embeddings, save_embeddings
from rag.llm import QwenClient
from rag.pipeline import answer_question, ensure_embeddings_device
from rag.retrieval import build_gold_mask, evaluate_hit_at_k


def embed_command(args: argparse.Namespace) -> None:
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


def eval_command(args: argparse.Namespace) -> None:
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


def ask_command(args: argparse.Namespace) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    enc_defaults = EncoderConfig()
    llm_defaults = LLMConfig()
    parser = argparse.ArgumentParser(
        description="CLI entrypoint for the modular TechQA RAG pipeline. See scripts/ for task-specific scripts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed = subparsers.add_parser("embed", help="Encode TechQA passages and save embeddings.")
    embed.add_argument("--embedding-path", type=Path, default=enc_defaults.embedding_path)
    embed.add_argument("--model", default=enc_defaults.model_name)
    embed.add_argument("--device", default=enc_defaults.device)
    embed.add_argument("--batch-size", type=int, default=enc_defaults.passage_batch_size)
    embed.add_argument("--max-length", type=int, default=enc_defaults.max_length)
    embed.set_defaults(func=embed_command)

    eval_parser = subparsers.add_parser("eval", help="Run batched Hit@K retrieval evaluation.")
    eval_parser.add_argument("--embedding-path", type=Path, default=enc_defaults.embedding_path)
    eval_parser.add_argument("--model", default=enc_defaults.model_name)
    eval_parser.add_argument("--device", default=enc_defaults.device)
    eval_parser.add_argument("--k", type=int, default=enc_defaults.k)
    eval_parser.add_argument("--batch-size", type=int, default=enc_defaults.query_batch_size)
    eval_parser.add_argument("--split", default="train")
    eval_parser.set_defaults(func=eval_command)

    ask = subparsers.add_parser("ask", help="Answer a single question with RAG.")
    ask.add_argument("--question", "-q", required=True)
    ask.add_argument("--k", type=int, default=enc_defaults.k)
    ask.add_argument("--embedding-path", type=Path, default=enc_defaults.embedding_path)
    ask.add_argument("--encoder-model", default=enc_defaults.model_name)
    ask.add_argument("--encoder-device", default=enc_defaults.device)
    ask.add_argument("--max-length", type=int, default=enc_defaults.max_length)
    ask.add_argument("--llm-model", default=llm_defaults.model_name)
    ask.add_argument("--llm-device", default=llm_defaults.device)
    ask.add_argument("--max-new-tokens", type=int, default=llm_defaults.max_new_tokens)
    ask.add_argument("--temperature", type=float, default=llm_defaults.temperature)
    ask.add_argument("--top-p", type=float, default=llm_defaults.top_p)
    ask.set_defaults(func=ask_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

