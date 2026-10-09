"""The task as loaded: a unit, its roster and the documents it may cite, embargo applied."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

TargetType = Literal["classification", "regression", "ranking"]
#: A target named as a change over the horizon (a move, growth or return), by whole words of its name; "return on"
#: (equity, assets) names a ratio, and "exchange" is not "change".
CHANGE = re.compile(r"(?<![a-z])(?:change|growth|returns?)(?![a-z])(?!_on(?![a-z]))")

#: The embargo compares dates as plain strings, which is only correct while both sides are
#: zero-padded ISO. All 69 public documents and all 9 cutoffs are -- but a held-out family
#: shipping "2024-2" or "2024-02" would silently misorder, and the failure is a stale citation
#: that costs the unit. Cheaper to refuse the date than to debug the comparison.
IsoDate = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]


class Doc(BaseModel):
    """A corpus document that has already cleared the embargo.

    `doc_date` is required, not optional. The toolkit counts an undated document as an embargo
    violation of its own (`qfbench2_track_analysis/corpus.py`) because a date-less document cannot
    be shown to predate the cutoff. Making the field required means one cannot exist downstream --
    `load_unit` drops it while it is still raw JSON, so nothing later has to remember the rule.
    """

    doc_id: str = Field(min_length=1)
    source: str = ""
    doc_date: IsoDate
    title: str = ""
    text: str
    #: Which roster entities the unit manifest says the document is about (`entity_ids`), or that it is market-wide
    #: (`shared`). Scorer 5.2.x counts a claim citing a document not about its entity, nor shared, as false. None
    #: when the manifest carries no label for it.
    entity_ids: list[str] | None = None
    shared: bool = False


class Entity(BaseModel):
    """One roster member. `features` is whatever the task file carried beyond the known keys."""

    entity_id: str = Field(min_length=1)
    name: str = ""
    #: The corpus belongs to the ENTITY, not the unit -- every row names its own. Identical
    #: across every public unit, but the schema allows per-entity corpora and a held-out family
    #: may use them.
    corpus_ref: str = "corpus/"
    #: Resolved at load, AFTER the embargo filter, so an embargoed document is absent here rather
    #: than present and unusable.
    doc_ids: list[str] = Field(default_factory=list)
    features: dict[str, object] = Field(default_factory=dict)


class Unit(BaseModel):
    """A task plus the corpus it is allowed to cite, with the embargo already applied."""

    task_id: str = Field(min_length=1)
    target_type: TargetType
    target_name: str
    target_labels: list[str] = Field(default_factory=list)
    #: What each label asserts about an entity (`target.label_assertions`, scorer 5.2.x units).
    label_assertions: dict[str, str] = Field(default_factory=dict)
    cutoff_date: IsoDate
    resolution_date: IsoDate | None = None
    interval_level: float = Field(default=0.90, gt=0.0, lt=1.0)
    prompt: str = ""
    entities: list[Entity] = Field(default_factory=list)
    #: The task table a `"task"` citation indexes (scorer 5.2.x, docs/CONCEPTS.md "Citing the task table"): one
    #: `json.dumps` line per task.json entities row, and each entity's row as [start, end).
    task_table: str = ""
    task_rows: dict[str, tuple[int, int]] = Field(default_factory=dict)
    docs: list[Doc] = Field(default_factory=list)

    @property
    def change_target(self) -> bool:
        return bool(CHANGE.search(self.target_name.lower()))

    @model_validator(mode="after")
    def _classification_has_a_vocabulary(self) -> Unit:
        # `alignment.py` requires an in-vocabulary label on classification. A unit that reaches
        # the model without one cannot produce a valid answer, and the failure would surface as
        # LABEL_INVALID at scoring time rather than here.
        if self.target_type == "classification" and not self.target_labels:
            raise ValueError(f"{self.task_id}: a classification unit needs target_labels")
        return self
