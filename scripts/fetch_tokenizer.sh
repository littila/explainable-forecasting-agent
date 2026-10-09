#!/usr/bin/env bash
# The model's own tokenizer (NVIDIA Nemotron 3 Super), the same file and checksum the Docker image downloads. The agent
# plans calls and fits documents in exact token counts; without this file it falls back to an estimate.
set -euo pipefail
URL=https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8/resolve/744b1880a37996c5d56bf454ae164dfd74d77c4e/tokenizer.json
SHA256=623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7
cd "$(dirname "$0")/.."
out=solution/tokenizer/tokenizer.json
if [ ! -f "$out" ]; then
  curl -fsSL "$URL" -o "$out.part"
  mv "$out.part" "$out"
fi
echo "$SHA256  $out" | shasum -a 256 -c -
