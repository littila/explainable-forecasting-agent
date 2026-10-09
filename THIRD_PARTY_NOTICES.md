# Third-party notices

This repository's own code is under the MIT License (`LICENSE`). It includes or depends on the following material,
each under its own terms.

## Included

| Path | Origin | License |
|---|---|---|
| `baselines/guardrails_example/citation_rail.py`, `baselines/guardrails_example/__init__.py` | [Agenthon-2026/track4-analysis-public](https://github.com/Agenthon-2026/track4-analysis-public) at `1c744e1`, unmodified. The agent runs the organisers' own reason and claim checks before it writes an answer. | MIT (text below) |
| `solution/analysis.schema.json` | The Track 4 answer schema from the shared toolkit `qfbench2-common` v2.5.1, [Agenthon-2026/Agenthon2026-public](https://github.com/Agenthon-2026/Agenthon2026-public), unmodified; a test checks it against the toolkit's copy. | CC BY-NC 4.0 |
| `solution/tokenizer/LICENSE`, `solution/tokenizer/NOTICE` | NVIDIA Nemotron 3 Super tokenizer licence and notice. The tokenizer file itself is not in this repository: the Docker build downloads it from Hugging Face by revision and checksum. | NVIDIA Nemotron Open Model License |

## Not included

- **The practice units** (task files and document corpora drawn from SEC EDGAR, FRED/ALFRED, BLS, CFTC, TreasuryDirect
  and other public sources) are the organisers' data. `scripts/fetch_units.sh` fetches them from the official Track 4
  repository.
- **Development dependencies**, installed by `uv sync`: the Track 4 scorer `qfbench2-track-analysis` (MIT) and the shared
  toolkit `qfbench2-common` (CC BY-NC 4.0). The tests check the agent's rules against their code.

## MIT License (Agenthon-2026/track4-analysis-public)

    MIT License
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
