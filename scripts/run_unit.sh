#!/usr/bin/env bash
# One unit through the submission entrypoint, as the harness runs it, against an OpenAI-compatible endpoint.
#   OPENROUTER_API_KEY=... scripts/run_unit.sh t4-fomc-curve-20240918 [out_dir]
# Defaults to the same model the House serves (NVIDIA Nemotron 3 Super) on OpenRouter's free route.
set -euo pipefail
unit=$1
out=${2:-runs/$unit}
mkdir -p "$out/traces"
MODEL_ENDPOINT=${MODEL_ENDPOINT:-https://openrouter.ai/api/v1} \
MODEL_NAME=${MODEL_NAME:-nvidia/nemotron-3-super-120b-a12b:free} \
T4_TRACE_DIR="$out/traces" PYTHONPATH=. \
  uv run python -m solution.cli analyze --task "units/$unit/task.json" --corpus "units/$unit/corpus/" \
  --out "$out/answer.json" | tee "$out/run.log"
