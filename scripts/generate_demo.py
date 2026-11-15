#!/usr/bin/env python3
"""Generate text with NeuroServe.

Usage (inside the dev container):
    PYTHONPATH=python python3 scripts/generate_demo.py "The capital of France is"
"""

import argparse
import os
import time

from tokenizers import Tokenizer

import neuroserve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--model-dir", default=os.environ.get("NS_MODEL_DIR", "/models/llama-3.2-1b-fp16"))
    parser.add_argument("--tokenizer", default=os.environ.get("NS_TOKENIZER", "/models/llama-3.2-1b/tokenizer.json"))
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    prompt_ids = tokenizer.encode(args.prompt).ids

    t0 = time.perf_counter()
    engine = neuroserve.Engine(args.model_dir, num_pages=512)
    t1 = time.perf_counter()
    out_ids = engine.generate(prompt_ids, max_new_tokens=args.max_new_tokens)
    t2 = time.perf_counter()

    print(args.prompt + tokenizer.decode(out_ids))
    print(
        f"\n[load {t1 - t0:.1f} s | {len(prompt_ids)} prompt tokens | "
        f"{len(out_ids)} new tokens in {t2 - t1:.2f} s ({len(out_ids) / (t2 - t1):.1f} tok/s)]"
    )


if __name__ == "__main__":
    main()