#!/usr/bin/env python3
"""Generate text with NeuroServe.

Usage (inside the dev container):
    PYTHONPATH=python python3 scripts/generate_demo.py "The capital of France is"
    PYTHONPATH=python python3 scripts/generate_demo.py --chat "What is the capital of France?"
"""

import argparse
import os
import time

import neuroserve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt")
    parser.add_argument("--chat", action="store_true", help="format the prompt as a user chat message")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--model-dir", default=os.environ.get("NS_MODEL_DIR", "/models/llama-3.2-1b-instruct-fp16"))
    parser.add_argument("--tokenizer-dir", default=os.environ.get("NS_TOKENIZER_DIR", "/models/llama-3.2-1b-instruct"))
    args = parser.parse_args()

    tok = neuroserve.ChatTokenizer(args.tokenizer_dir)
    if args.chat:
        prompt_ids = tok.encode_chat([{"role": "user", "content": args.prompt}])
    else:
        prompt_ids = tok.encode(args.prompt)

    t0 = time.perf_counter()
    engine = neuroserve.Engine(args.model_dir, num_pages=512)
    t1 = time.perf_counter()
    out_ids = engine.generate(prompt_ids, max_new_tokens=args.max_new_tokens)
    t2 = time.perf_counter()

    # The engine has no stop handling yet: cut at the first stop token.
    stop_at = next((i for i, t in enumerate(out_ids) if t in tok.stop_token_ids), len(out_ids))
    kept = out_ids[:stop_at]

    # Stream the answer token by token, the way the API server will.
    detok = tok.detokenizer()
    print(args.prompt if not args.chat else f"User: {args.prompt}\nAssistant:", end="", flush=True)
    for token_id in kept:
        print(detok.push(token_id), end="", flush=True)
    print(detok.flush())

    stopped = "stop token" if stop_at < len(out_ids) else "max tokens"
    print(
        f"\n[load {t1 - t0:.1f} s | {len(prompt_ids)} prompt tokens | {len(kept)} tokens kept ({stopped}) | "
        f"{len(out_ids)} generated in {t2 - t1:.2f} s ({len(out_ids) / (t2 - t1):.1f} tok/s)]"
    )


if __name__ == "__main__":
    main()
