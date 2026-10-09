"""Token counts for budgets, in the model's own tokenizer when it is available.

Characters per token run from 1.8 (numeric tables) to 3.5 (prose) on the rendered corpus, so a
fixed ratio misjudges a budget by up to 2x. Exact counting reads the Nemotron-3-Super
`tokenizer.json` from `T4_TOKENIZER` or the local Hugging Face cache. Without it, a fit on digits,
words and other characters is used (-37%..+12% against exact on 776 rendered sections), scaled up
so that it undercounts in fewer than 5% of them.
"""

from __future__ import annotations

import glob
import os
import pathlib
import re
from collections.abc import Callable

MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
_FIT = (1.7126, -0.4207, 0.2576)
_FIT_FLOOR = 0.8


def _tokenizer_path() -> str:
    if path := os.environ.get("T4_TOKENIZER"):
        return path
    cache = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".cache/huggingface")) / "hub"
    found = glob.glob(str(cache / f"models--{MODEL.replace('/', '--')}/snapshots/*/tokenizer.json"))
    return found[0] if found else ""


def estimate(text: str) -> int:
    digits = sum(ch.isdigit() for ch in text)
    words = len(re.findall(r"[A-Za-z]+", text))
    return int((_FIT[0] * digits + _FIT[1] * words + _FIT[2] * (len(text) - digits)) / _FIT_FLOOR) + 1


def counter() -> Callable[[str], int]:
    path = _tokenizer_path()
    if path and os.path.exists(path):
        try:
            from tokenizers import Tokenizer
        except ImportError:
            return estimate
        tok = Tokenizer.from_file(path)
        return lambda text: len(tok.encode(text, add_special_tokens=False).ids)
    return estimate
