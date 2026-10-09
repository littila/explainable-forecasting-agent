"""Loading a unit: `task.json`, its roster and its corpus, with the embargo applied on raw JSON."""

from __future__ import annotations

import json
import pathlib

from solution.types import Doc, Entity, Unit


def _read_doc(path: pathlib.Path, cutoff: str) -> Doc | None:
    """Parse a corpus file, or return None if the embargo excludes it.

    The filter is here, on raw JSON, rather than on a constructed `Doc`: `Doc.doc_date` is a
    required ISO date, so an embargoed document never becomes an object at all.
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    date = d.get("doc_date")
    if not date or date > cutoff:
        return None  # undated is a violation in its own right, not a permissive default
    # the scorer resolves a citation by the file's name, and reads a document's text from `text`, else its spans'
    # texts joined by a space
    text = d.get("text") if isinstance(d.get("text"), str) else " ".join(
        s.get("text", "") for s in d.get("spans") or [] if isinstance(s, dict) and isinstance(s.get("text"), str))
    return Doc(
        doc_id=path.stem,
        source=d.get("source") or "",
        doc_date=date,
        title=d.get("title") or "",
        text=text,
    )


def _resolve(unit_dir: pathlib.Path, corpus_ref: str,
             corpus_dir: pathlib.Path | None = None) -> list[pathlib.Path]:
    """A `corpus_ref` is a directory of documents or a single document, relative to the unit.

    `corpus_dir` is the published CLI's `--corpus`, which the harness passes separately from
    `--task`. It overrides the unit-relative resolution ONLY for a plain directory ref, so an
    entity that scopes itself to its own file or subdirectory still lands inside the corpus the
    harness named rather than silently ignoring it.
    """
    ref = corpus_ref.strip().lstrip("/")
    if corpus_dir is not None:
        return _resolve(corpus_dir.parent, corpus_dir.name) if ref.rstrip("/") in ("corpus", "") \
            else _resolve(corpus_dir, ref.removeprefix("corpus/"))
    target = unit_dir / ref
    if target.is_dir():
        return [p for p in sorted(target.glob("*.json")) if p.name != "manifest.json"]
    return [target] if target.is_file() else []


def task_table(rows: list[dict]) -> tuple[str, dict[str, tuple[int, int]]]:
    """The task table a `"task"` citation indexes, as the scorer builds it (`task_table_text`): each entities row as
    one-line JSON in file key order, joined by newlines, and each entity's row as [start, end)."""
    lines, ranges, offset = [], {}, 0
    for row in rows:
        line = json.dumps(row, ensure_ascii=False, separators=(", ", ": "))
        ranges[row.get("entity_id", "")] = (offset, offset + len(line))
        lines.append(line)
        offset += len(line) + 1
    return "\n".join(lines), ranges


def labels(unit_dir: pathlib.Path) -> dict[str, tuple[list[str] | None, bool]]:
    """Each corpus document's label from the unit manifest, as the scorer reads it (`entity_ids`, `shared`), by
    doc_id (the file's stem); from `corpus/manifest.json` when the unit has no root manifest."""
    for path in (unit_dir / "manifest.json", unit_dir / "corpus" / "manifest.json"):
        if path.is_file():
            files = json.loads(path.read_text(encoding="utf-8")).get("files") or []
            return {pathlib.Path(f["path"]).stem: (list(f["entity_ids"]) if "entity_ids" in f else None,
                                                  bool(f.get("shared")))
                    for f in files if isinstance(f, dict) and f.get("role") == "corpus" and f.get("path")}
    return {}


def load_unit(unit_dir: str | pathlib.Path,
              corpus_dir: str | pathlib.Path | None = None) -> Unit:
    """Load a unit with the embargo already applied and each entity scoped to its own documents.

    **The embargo is applied here, at load**, so no later component can see a post-cutoff document
    even by mistake. The rule matches the toolkit's own (`qfbench2_track_analysis/corpus.py`):
    keep `doc_date <= cutoff_date`, and drop a document with no `doc_date` at all -- the toolkit
    counts an undated document as an embargo violation, because a date-less document cannot be
    shown to predate the cutoff. Citing one fails the unit.

    **Each document carries its manifest label** (`labels`): the entities it is about, or market-wide. That decides
    which documents an entity's claims may cite (`render.owned`).

    **The corpus belongs to the entity, not the unit.** Every entity carries its own `corpus_ref`.
    All eleven public units set it to "corpus/" for every row, so the distinction is invisible
    here -- which is exactly why it is worth honouring now rather than discovering on a held-out
    family that scopes each entity to its own subdirectory.
    """
    unit_dir = pathlib.Path(unit_dir)
    task = json.loads((unit_dir / "task.json").read_text(encoding="utf-8"))
    target = task.get("target", {})
    cutoff = str(task["cutoff_date"])[:10]

    table, rows = task_table(task.get("entities", []))
    marks = labels(unit_dir)
    cache: dict[pathlib.Path, Doc | None] = {}
    entities: list[Entity] = []
    docs: dict[str, Doc] = {}
    for raw in task.get("entities", []):
        ref = raw["corpus_ref"] if isinstance(raw.get("corpus_ref"), str) and raw["corpus_ref"].strip() else "corpus/"
        where = pathlib.Path(corpus_dir) if corpus_dir else None
        # a ref that names nothing in the corpus (a mount path, "./corpus/", a glob) reads the whole corpus: which
        # documents an entity may cite is the manifest's to say, not the ref's
        doc_ids = []
        for path in _resolve(unit_dir, ref.removeprefix("./"), where) or _resolve(unit_dir, "corpus/", where):
            if path not in cache:
                cache[path] = _read_doc(path, cutoff)
            doc = cache[path]
            if doc is None:
                continue  # embargoed: undated or post-cutoff
            docs[doc.doc_id] = doc
            doc_ids.append(doc.doc_id)
        entities.append(Entity(
            entity_id=raw.get("entity_id", ""),
            name=raw.get("name") or "",
            corpus_ref=ref,
            doc_ids=doc_ids,
            features={k: v for k, v in raw.items()
                      if k not in ("entity_id", "name", "corpus_ref")},
        ))

    # BOTH published shapes. SUBMISSION_CLI.md's interface table and contract invariant 7 put
    # `target_type` at the TOP LEVEL; all eleven public units nest it at `target.type`. Reading
    # only the nested one silently loads a top-level regression unit as a classification unit,
    # which is TARGET_TYPE_MISMATCH and W = -0.27 -- and the practice set cannot show it, because
    # no unit here uses the documented shape. The documented field wins where both are present.
    return Unit(
        task_id=task.get("task_id", unit_dir.name),
        target_type=task.get("target_type") or target.get("type") or "classification",
        target_name=target.get("name") or task.get("target_name") or "outcome",
        target_labels=list(target.get("labels") or task.get("target_labels") or target.get("label_assertions") or []),
        label_assertions=dict(target.get("label_assertions") or {}),
        cutoff_date=cutoff,
        resolution_date=str(task.get("resolution_date") or "")[:10] or None,
        interval_level=float(task.get("interval_level", 0.90)),
        prompt=task.get("prompt", ""),
        entities=entities,
        task_table=table,
        task_rows=rows,
        docs=[docs[k].model_copy(update=dict(zip(("entity_ids", "shared"), marks[k]))) if k in marks else docs[k]
              for k in sorted(docs)],
    )
