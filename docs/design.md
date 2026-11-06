# Design

The agent runs once per unit in a container. Code works out everything the documents settle on their own (forecasts
from tables, levels copied from filings, labels a filing states plainly), shows those decisions to the language model,
asks the model for the rest, and assembles an answer that no mechanical gate can refuse. This page covers the
principles, the pipeline step by step, and each component with its constants and measured effect. Measurements are
changes in the 10-unit mean of a local estimate (reconstructed outcomes, stand-in baselines) unless stated otherwise.

## Principles

1. **Data first; the model where it measurably wins.** When a unit's documents hold an entity's history as a dated
   table, code backtests a small menu of standard methods on it. The data's point replaces the model's only where it
   measured far better (revisions, growth from a copied level); elsewhere the model gives the point and the data gives
   the band. Without priors the mean drops 0.037; a 50/50 blend of model and prior points lost to the model's own
   point by 0.002, so it was replaced by the rule above.
2. **Every number is traceable.** A submitted number either comes from a computation on named table rows, and cites
   the last row it used, or is the model's, next to quotes that code places verbatim in the original text. A quote
   that cannot be found is not cited; a claim's text is the passage's own words, never a paraphrase.
3. **The reasons explain the submitted answer.** Code decides first and writes the decision on the entity's line as a
   *decided forecast*. The model's own block then agreed with the submitted value 36/36 (macro revisions), 23/24 (EPS
   growth) and 12/12 (credit events), against 19/36, 1/24 and 10/12 before.
4. **Simple code, documented choices.** Every component has an ablation row. Components within run-to-run noise were
   removed: corrective turns on answer calls, asking again, a level guard, a blend weight, and a 2,133-line document
   segmentation stack. The code shrank 44% and the score held.
5. **Triggered by data shape, never by unit names.** A component fires when the data has its shape, matched by whole
   words: a vintage table, a dated column named like the entity, a numeric `prior_*` base that names a level, a credit
   label set, a yield target in basis points. A shape the practice units lack is handled rather than refused.
6. **There is always an answer.** A valid answer is written before any model call; even a unit that fails to load gets
   one that passes the schema. Deadlines come from the unit's time limit; no unit makes more than 25 requests.
7. **Nothing learned from outcomes inside the image.** Methods are chosen per run by backtests on the unit's own
   pre-cutoff history; the few fixed constants are stated with their source in [provenance.md](provenance.md).

## Pipeline

| # | Step | Module | Who | What |
|---|---|---|---|---|
| 1 | Load | `corpus` | code | Task, roster, manifest labels; every document dated after the cutoff, or undated, is dropped at load. |
| 2 | Floor answer | `cli` | code | A valid answer from priors and fallbacks, written before any request. |
| 3 | Priors | `data`, `forecast`, `priors` | code | Dated table rows become each entity's series; a backtested forecast menu gives a point, a 90% band and the row it rests on. |
| 4 | Copy latest levels | `extract` | model | Growth targets only: the model copies each entity's latest reported level with a quote; code verifies it and computes a growth prior that decides the point. |
| 5 | Filing signals | `signals` | code | Credit-event labels only: going-concern doubt decides `credit_event`; an investment-grade rating with no speculative grade decides `no_event`. |
| 6 | Plan | `plan`, `render`, `prompt` | code | Entities sharing their own documents go together, at most 6 per call, each prompt within 40% of a 262K-token window; documents are fitted smaller when a group or the request budget needs it. |
| 7 | Ask | `pipeline`, `llm` | model | Answer calls drawn 5, 3 or 1 times; up to 4 calls in flight; an entity with no reply and no prior is asked once more. |
| 8 | Copy earlier levels | `extract` | model | Yield-change targets only: earlier yields stated in dated sentences, for the interval floor. |
| 9 | Read and merge | `reply` | code | Per entity, the median sample of the majority label, with an interval spanning those samples. |
| 10 | Combine | `priors`, `signals`, `moves` | code | The two rules (point, band), decided labels, the yield floor. |
| 11 | Cite and assemble | `cite`, `assemble` | code | Quotes placed in the original text; one valid row per entity; fallbacks for rows nothing answered. |
| 12 | Reasons and gates | `reply`, `preflight` | code | Up to 3 reasons the grader would judge, chosen against a clock; the scorer's mechanical gates run locally. |

## Components

### Load and the embargo (`corpus.py`, `types.py`)

Builds the unit: target name, type and labels, cutoff and resolution dates, entities with their features, documents
with their manifest labels (`entity_ids` or `shared`), and the task table as JSON for `"task"` citations. A document
dated after the cutoff is never constructed, so the stale-evidence trap of the Final cannot reach retrieval,
prompting or citation. The loader also accepts shapes the scorer allows but the practice units do not use: a label
vocabulary given only as definitions, document text in spans, null names, dates with a time part, a `corpus_ref` that
names nothing, a null `entity_ids` label. A document's id is its file name, which is how the scorer resolves
citations; with a manifest, a corpus file it does not declare is not loaded, since citing it would refuse the unit.

### The view (`render.py`, `retrieve.py`, `tokens.py`)

Each document is shown as its original text in a `<doc>` tag with its title, date, source and manifest label,
forward-dated rows included (in a document dated before the cutoff they are calendars or projections, known on the
document's date). A document over its cap has its whitespace collapsed; if still too big it is cut into windows of
about 16,000 characters (or half its cap in characters when that is smaller, never under 200, so a small cap still
shows the best passage), ranked by BM25 against a query built from the task, and the best windows that fit are shown
in document order with gap markers. Token counts are exact, in the model's own tokenizer.

| Constant | Value |
|---|---|
| `WINDOW`, `CONTEXT_SHARE` | 262,144 tokens, 0.4 |
| `DOC_MAX` | 96,665 tokens per document (smaller when the plan sets a cap) |
| `WINDOW_CHARS`, `MIN_WINDOW_CHARS` | 16,000, 200 |

Head-and-tail truncation kept 0 of 9 going-concern passages in one issuer's 10-K; BM25 kept all 9. Replacing an
earlier segmentation into numbered lines with plain text left all 57 priors unchanged, cut prompts 17 to 25% on table
units and raised the estimate from 0.646 to 0.659.

### Tables as data (`data.py`)

A table is a header line of cells followed by rows whose first cell is a date, separated by `|` or tabs (a Markdown
row's outer pipes dropped). The embargo is the document's date: every row of an admitted document is read, as the
model reads it. Every value keeps its row's span, so a
computation cites exactly the rows it used. Two views: one column's series in date order, and a vintage table (columns
`as_of_<release>`) giving each reference period's estimate at each release.

### The forecast menu (`forecast.py`)

- **Next value:** last value, mean of the last 3 or 6, mean or median of all, each backtested one step ahead on the
  series' own history; the smallest mean absolute error wins, and the band is the point plus that method's error
  quantiles.
- **Change** to the resolution date, counted in the series' own steps (business days, else whole periods of its median
  gap; a monthly row stands for its month's last day): *no change* against *drift*, backtested at that horizon when
  there are 5 origins, else no change with the step volatility scaled by the square root of the horizon.
- **Revision** of a reference month: the latest estimate plus the median of past revisions over the same number of
  releases at the same age, leaving out comprehensive updates (over 10 times the usual revision).

Intervals: from 19 backtest errors up, their 5% and 95% quantiles; with fewer, `mean +- t(n-1) s sqrt(1 + 1/n)`.

### Priors and the two rules (`priors.py`)

A series is found from the data: a vintage table in the entity's own documents; else a dated column named as the
entity (its name or a text value of its row such as a series id); else the one column of its own documents whose words
all appear in the target's name. A change target (the whole word change, growth or return) is forecast as a change,
in basis points when the target says so and the series is in percent. **Rule 1:** the point is the model's, unless the
prior decides it. **Rule 2:** the interval is the prior's band placed around that point. Every prior cites its anchor
row. An entity whose series no method can read gets no prior; the others keep theirs.

### Copies (`extract.py`)

Small model calls that copy printed figures and forecast nothing ("You copy figures from documents. You do not
forecast, estimate or compute."). For a growth target, each entity's latest reported level is kept only when the value
is one of its quote's numbers and within a fifth to five times the base, the quote is found in the entity's own
documents, and the period does not end after those documents' date; growth is `(level - base) / |base|`. For a
yield-change target, a second kind lists earlier yields stated in dated sentences. A copy that fails the checks is sent
back once. Copies take at most 12 of the 25 requests. Without the growth copy, EPS-growth MAE rises from 8.4 to 21.9.

### Filing signals (`signals.py`)

For the credit-event label set only: "substantial doubt about … ability to continue as a going concern", not negated,
decides `credit_event`; an investment-grade rating with no speculative grade in any rating passage decides `no_event`.
The decided label goes on the entity's line before the call. Without it, credit-event accuracy falls from 0.833 to
0.750.

### Yield floor (`moves.py`)

For a yield-change target (a yield in basis points: `bps`, `bp` or `basis_points` in its name), a least half-width
equal to the largest move the documents state between an earlier level and the cutoff level, scaled to the horizon
by the square root of time; none when no earlier level is stated. No constant is fitted: an earlier form added fitted
constants (28 bps, a factor 1.05) from past FOMC meetings, which the organisers' ruling on held-out cutoffs ruled out
for the Final. The model writes about +-10 bps for a yield change whatever the documents say; the floor turned 0 of 6
covered into 6 of 6 on one unit.

### Plan (`plan.py`)

The fewest calls: entities that share their own documents together; at most 6 per call by the output budget
(`(0.85 x 4,000 - 600) / 460`). A group whose documents exceed the call (two large filings, a filing about two
entities) gets the largest shared cap at which it fits; with a request budget, every document is lowered to the
largest cap at which the plan fits that many calls. A call shows at most its budget over 1,000 tokens of documents
(`MIN_DOC`), an entity with more showing its latest that many (the rest stay citable), and a unit no plan fits is
asked with its entity rows alone rather than answered from its floor.

### Prompt (`prompt.py`)

System: a careful financial analyst; only the documents and the entity rows ("where the documents differ from what you
remember, the documents are right"); verbatim quotes; what the target means; what a good answer is; the output format
with a filled sample. User: the documents; the task's facts (target, each label's definition, cutoff and resolution
dates) and its prompt; the entity lines with any decided forecast; the entities to answer in order. Replies are
structured text (`<reasons>` then `<forecasts>`), parsed by code: equal in quality to provider JSON in a head-to-head,
and cheaper. Stating the label definitions fixed a borderline post-earnings label (accuracy 0.667 to 1.000).

### Calls and samples (`pipeline.py`, `llm.py`)

Temperature 0, a constant seed (verification reruns get a fresh `QFBENCH_SEED`), thinking off on the House route. Rate limits and server errors are retried with backoff;
after three unreachable calls in a row the unit is answered from what is in hand. At most 4 calls and 200,000 prompt
tokens in flight; starts and corrections draw on one locked request budget. Answer calls are drawn 5 times when that
fits 15 requests and what the copies left, else 3, else once; samples are worth 0.014 on the mean.

### Reading, merging, citing, assembling (`reply.py`, `cite.py`, `assemble.py`)

Replies are parsed line by line; per entity the merge keeps the majority label, the median sample's block and an
interval spanning those samples (worth 0.008). Quotes are placed only in documents the entity may cite, folding case,
quote marks, dashes and whitespace; whole quotes before openings, the longest opening found by bisection. Assembly
gives one valid row per entity: labels matched to the vocabulary, a missing point from the floor, a row nothing
answered given the answered rows' median width and most common label, duplicates and content-free claims dropped, and
a citation for every row (its own task row as the last resort).

### Reasons (`reply.selected`)

Up to 3 reasons, ranked (cited first, broadest scope, earliest in their call) and then checked one at a time with the
organisers' own rules (`baselines/guardrails_example/citation_rail.py`) until 3 are kept or the clock runs out. Each
check reads only the documents the reasons cite: checking every reason against the whole corpus had made this the
slowest stage on filing units (replaying 30 runs: 6 minutes 12 seconds before, 20 seconds after, same reasons chosen).

### Entry, clock and checks (`cli.py`, `preflight.py`)

`analyze --task /input/task.json --corpus /input/corpus/ --out /output/answer.json`. The floor answer is written first
and replaced at the end; the hard deadline is the card's time limit less 30 seconds (570 s on a 600 s card), with a soft
deadline 30 s earlier and an alarm at the hard one. `preflight` runs the scorer's mechanical gates locally, and a
reasons block that would fail the schema is removed rather than risk the unit.
