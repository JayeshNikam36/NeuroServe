#!/usr/bin/env python3
"""Convert a Hugging Face Llama checkpoint into cuda-llm-engine's weight format.

Writes weights.bin + index.json + index.tsv in exactly the format of cuda-llm-engine's
export_for_cpp.py and make_index_tsv.py (sorted tensor names, FP16, byte offsets), but with
the input and output folders as arguments. Run on the host (needs torch and safetensors):

    ~/venvs/vllm/bin/python scripts/export_weights.py \
        --model-dir ~/models/llama-3.2-1b-instruct --out-dir ~/models/llama-3.2-1b-instruct-fp16
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

FP16_MAX = 65504.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(model_dir.glob("*.safetensors"))
    if len(files) != 1:
        raise SystemExit(f"expected exactly one .safetensors file in {model_dir}, found {len(files)}")

    index, offset = [], 0
    with safe_open(str(files[0]), "pt") as f, open(out_dir / "weights.bin", "wb") as out:
        for name in sorted(f.keys()):
            tensor = f.get_tensor(name)
            peak = float(tensor.abs().max())
            if peak > FP16_MAX:
                raise SystemExit(f"{name}: max |value| {peak} does not fit in FP16")
            t16 = tensor.to(torch.float16).contiguous()
            data = t16.numpy().tobytes()
            out.write(data)
            index.append({"name": name, "shape": list(t16.shape), "dtype": "f16",
                          "offset": offset, "nbytes": len(data)})
            offset += len(data)

    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    with open(out_dir / "index.tsv", "w") as f:
        for t in index:
            f.write(f"{t['name']}\t{t['offset']}\t{t['nbytes']}\t{','.join(map(str, t['shape']))}\n")
    print(f"{len(index)} tensors, {offset / 1e9:.3f} GB -> {out_dir}")


if __name__ == "__main__":
    main()
