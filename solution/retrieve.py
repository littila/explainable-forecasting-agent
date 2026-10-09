"""BM25 over a document's windows (`render.windows`), for a document too large to show whole.

The query is built from the task file alone, so the view is deterministic. The judge scores a
citation by whether its span entails our prediction, which BM25 only correlates with.
"""

from __future__ import annotations

import math
import re

from solution.types import Unit

_TOKEN = re.compile(r"[a-z0-9]+")

#: Words with no discriminating power in a financial corpus. Short on purpose: a long hand-tuned
#: list overfits to the families we have seen.
_STOP = frozenset((
    "a", "an", "the", "of", "for", "to", "in", "on", "at", "by", "and", "or", "is", "are", "was",
    "were", "be", "been", "it", "its", "this", "that", "these", "those", "with", "from", "as",
    "we", "our", "you", "your", "will", "would", "may", "might", "can", "could", "shall",
    "should"))

#: Words every task prompt carries and no document discusses: the instructions, not the target.
_INSTRUCTION = frozenset(re.findall(r"\S+", """using only frozen evidence corpus each available before
cutoff predict for in the table its of relative same one earlier is given provide and to that around it
support your reasoning citations passages most recent any do not rely post absent by design reported
market after expressed as from quarter ended year point forecast interval percent sec filings pre
drivers reason"""))
#: Abbreviations a target name uses where the documents spell the concept out.
_EXPAND = {"eps": "diluted earnings per share net income", "yoy": "year over year", "pct": "percent"}


def _terms(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1 and t not in _STOP]


class BM25:
    """Okapi BM25, each text a document."""

    def __init__(self, texts: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf: list[dict[str, int]] = []
        self.df: dict[str, int] = {}
        for text in texts:
            counts: dict[str, int] = {}
            for term in _terms(text):
                counts[term] = counts.get(term, 0) + 1
            self.tf.append(counts)
            for term in counts:
                self.df[term] = self.df.get(term, 0) + 1
        self.length = [sum(c.values()) for c in self.tf]
        self.avg_len = sum(self.length) / len(self.length) if self.length else 0.0

    def _idf(self, term: str) -> float:
        n, df = max(len(self.tf), 1), self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> list[float]:
        out = [0.0] * len(self.tf)
        terms = _terms(query)
        if not terms or not self.avg_len:
            return out
        for term in terms:
            idf = self._idf(term)
            for i, counts in enumerate(self.tf):
                f = counts.get(term, 0)
                if f:
                    norm = 1 - self.b + self.b * (self.length[i] / self.avg_len)
                    out[i] += idf * (f * (self.k1 + 1)) / (f + self.k1 * norm)
        return out

    def ranked(self, query: str) -> list[int]:
        """Indices of the texts that match at all, best first, ties by position."""
        scored = self.score(query)
        return sorted((i for i, v in enumerate(scored) if v > 0), key=lambda i: (-scored[i], i))


def task_terms(unit: Unit) -> str:
    """The target concept and what the task names as mattering, instruction words removed."""
    words = re.findall(r"[a-z][a-z-]+", f"{unit.target_name.replace('_', ' ')} {unit.prompt}".lower())
    kept = [_EXPAND.get(w, w) for w in words if w not in _INSTRUCTION and len(w) > 2]
    return " ".join(dict.fromkeys(kept))


def retrieval_query(unit: Unit) -> str:
    # the target's name and prompt, never the entity: in an entity's own documents every node names it
    plain = " ".join(re.split(r"[_\W]+", f"{unit.target_name} {unit.prompt[:400]}"))
    return f"{task_terms(unit)} {plain}"
