from typing import Any, Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import LLMConfig


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful technical support assistant. "
    "Explain the root cause briefly and then give the concrete workaround or commands "
    "the user should run, based ONLY on the documents. "
    "If the documents do not contain a clear solution, say you are not sure."
)


class QwenClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(cfg.device)
        self.model.eval()

    @staticmethod
    def format_contexts(retrieved: Iterable[dict[str, Any]]) -> str:
        chunks = []
        for i, r in enumerate(retrieved, start=1):
            chunks.append(f"[DOC {i}] (score={r['score']:.3f})\n{r['text']}")
        return "\n\n".join(chunks)

    def build_messages(
        self, question: str, retrieved: List[dict[str, Any]], system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ) -> List[dict[str, str]]:
        context_block = self.format_contexts(retrieved)
        user_content = f"""Question:
{question}

Relevant documents:
{context_block}

Answer the question in your own words. If you use a document, mention its DOC number."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def generate(self, messages: List[dict[str, str]]) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

