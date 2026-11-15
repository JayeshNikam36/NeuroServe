#!/usr/bin/env python3
"""Write Hugging Face reference token ids for chat-template and plain-text encoding.

Run on the host (needs transformers):
    ~/venvs/vllm/bin/python scripts/make_chat_reference.py \
        --tokenizer-dir ~/models/llama-3.2-1b-instruct \
        --out ~/models/llama-3.2-1b-ref/chat_reference.json
"""

import argparse
import json
from pathlib import Path

import transformers
from transformers import AutoTokenizer

DATE = "26 Jul 2024"  # pinned so the template's "Today Date" line is reproducible

CHAT_CASES = [
    {"name": "single_user", "add_generation_prompt": True,
     "messages": [{"role": "user", "content": "What is the capital of France?"}]},
    {"name": "system_and_user", "add_generation_prompt": True,
     "messages": [{"role": "system", "content": "You are a concise assistant."},
                  {"role": "user", "content": "Explain CUDA memory coalescing in one sentence."}]},
    {"name": "multi_turn", "add_generation_prompt": True,
     "messages": [{"role": "user", "content": "Hi!"},
                  {"role": "assistant", "content": "Hello! How can I help?"},
                  {"role": "user", "content": "Name three GPU memory types."}]},
    {"name": "unicode", "add_generation_prompt": True,
     "messages": [{"role": "user", "content": "Translate to French: naïve café, 東京, 🎉"}]},
    {"name": "whitespace_is_trimmed", "add_generation_prompt": True,
     "messages": [{"role": "user", "content": "   padded question?  \n"}]},
    {"name": "no_generation_prompt", "add_generation_prompt": False,
     "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]},
]

PLAIN_TEXTS = ["The capital of France is", "def fibonacci(n):", "naïve café 東京 🎉"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(str(Path(args.tokenizer_dir).expanduser()))
    out = {"transformers_version": transformers.__version__, "date_string": DATE, "chat": [], "plain": []}

    for case in CHAT_CASES:
        # Exactly what apply_chat_template(tokenize=True) does internally: render, then encode
        # without adding special tokens (the template already contains <|begin_of_text|>).
        text = tok.apply_chat_template(case["messages"], add_generation_prompt=case["add_generation_prompt"],
                                       tokenize=False, date_string=DATE)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        out["chat"].append({**case, "text": text, "ids": ids})

    for text in PLAIN_TEXTS:
        out["plain"].append({"text": text, "ids": tok(text)["input_ids"]})

    path = Path(args.out).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(out['chat'])} chat cases and {len(out['plain'])} plain cases to {path}")


if __name__ == "__main__":
    main()
