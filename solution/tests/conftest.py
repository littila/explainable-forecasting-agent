import os
import pathlib

# The tests pin plans and fits measured in the model's own token counts: use the tokenizer that
# scripts/fetch_tokenizer.sh puts beside the code, as the image does.
TOKENIZER = pathlib.Path(__file__).resolve().parents[1] / "tokenizer" / "tokenizer.json"
if TOKENIZER.exists():
    os.environ.setdefault("T4_TOKENIZER", str(TOKENIZER))
