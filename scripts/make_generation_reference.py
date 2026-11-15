#!/usr/bin/env python3
"""Hugging Face greedy outputs for the chat-reference prompts (run on the host).

Uses the exact prompt ids from chat_reference.json, so both sides see identical input.

    ~/venvs/vllm/bin/python scripts/make_generation_reference.py \
        --model-dir ~/models/llama-3.2-1b-instruct \
        --chat-ref ~/models/llama-3.2-1b-ref/chat_reference.json \
        --out ~/models/llama-3.2-1b-ref/instruct_greedy.json
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--chat-ref", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser()
    ref = json.loads(Path(args.chat_ref).expanduser().read_text(encoding="utf-8"))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float16).cuda().eval()
    eos = model.generation_config.eos_token_id
    pad = eos[0] if isinstance(eos, list) else eos

    cases = []
    for case in ref["chat"]:
        if not case["add_generation_prompt"]:
            continue  # nothing to generate after a finished assistant turn
        ids = torch.tensor([case["ids"]], device="cuda")
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=pad)
        new_ids = out[0, ids.shape[1]:].tolist()
        cases.append({"name": case["name"], "prompt_ids": case["ids"], "greedy_ids": new_ids})
        print(f"{case['name']}: {len(new_ids)} tokens")

    path = Path(args.out).expanduser()
    path.write_text(json.dumps({"model_dir": str(model_dir), "max_new_tokens": args.max_new_tokens,
                                "cases": cases}, indent=1), encoding="utf-8")
    print(f"wrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main()
