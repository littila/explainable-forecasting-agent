# Data and artifact provenance

The competition requires every submission to document what its image is built from and how it respects each
task's cutoff (the Track 4 repository's `docs/TRAINING-POLICY.md` and `docs/ARTIFACT-POLICY.md`). This is that record,
as filed with the submission; paths refer to this repository.

## Summary

This record describes what the Track 4 submission image is built from, as the training and artifact policies
require. The image holds no learned artifacts and no training data: every forecast comes
from the House model through the API, combined at run time with statistics computed from the unit's own pre-cutoff
history, and the image adds only pinned Python packages, the model's tokenizer file, the organizers' reason checker and
our own code. At evaluation time it reads only the supplied task and its frozen corpus; the BM25 index and the
statistics are computed per run over that corpus and never stored. No number in the image was fitted to outcomes;
design choices were compared during Development on the ten practice units' realized outcomes, which the organizers'
ruling on issue #24 permits with the disclosure in "Practice-unit outcomes" below (the practice units are named there as
the in-sample exception). The submitted copy is documentation for organizer verification, not a descriptor field, and is
not added to the upload ZIP.

## Image

| item | value |
|---|---|
| latest image | `ghcr.io/littila/t4-solution:c5-6d427df` (commit 6d427df), linux/amd64 manifest `sha256:e42acd80c283d3544cf9354b588d3fc72b78cc8e29561d41594174d3d89b2ca8` (`submission.json`) |
| built from | this repository at commit `618228e`, `Dockerfile` |
| base image | `python:3.13-slim` (Docker Hub, by tag) |
| Python packages | `solution/requirements.txt`: openai 3.16.2, pydantic 2.13.5, jsonschema 4.26.0, tokenizers 0.23.2 (exact pins; their dependencies resolved by pip at build) |

## Components

| component | kind | source, revision, checksum | license |
|---|---|---|---|
| `solution/` | our code: rendering, planning, prompts, parsing, citation resolution, assembly | this repository | repository `LICENSE` |
| `solution/tokenizer/tokenizer.json` | static asset: tokenizer vocabulary only, no weights; counts prompt tokens | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` at revision `744b1880a37996c5d56bf454ae164dfd74d77c4e`, sha256 `623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7` (checked by the `Dockerfile`) | NVIDIA Nemotron Open Model License (`solution/tokenizer/LICENSE`, `NOTICE`) |
| `solution/analysis.schema.json` | static asset: answer schema | vendored from the toolkit `qfbench2-common` v2.5.1; a test asserts it equals the installed copy | toolkit license |
| `baselines/guardrails_example/citation_rail.py` | organizers' code: the reason checks (`check_submitted_reasons`, `reasons_judged`), standard library only on that path | this repository (organizer baseline) | repository `LICENSE` |
| BM25 index (`solution/retrieve.py`) | built at run time over the unit's own frozen corpus to fit long filings; never stored | - | - |

## Run-time statistics (`solution/data.py`, `solution/forecast.py`, `solution/priors.py`)

For an entity whose own history is a dated series in its unit's corpus, each run computes a prior from that history
alone: rows and release columns dated after the cutoff are dropped first. A fixed menu of standard methods (last
value, mean of the last 3 or 6, mean or median of all; for a change, no change or drift; for a vintage table, the
median past revision at the same age) is backtested on the series' own pre-cutoff history and the smallest backtest
error picks the point. The interval comes from that method's backtest errors. Nothing is fitted offline, stored, or
chosen on outcomes; the model's point is combined with the prior at run time.

| constant | value | how it was set |
|---|---|---|
| interval quantiles | (1 - level) / 2 and its complement | the interval's definition (5% and 95% for 90%) |
| empirical quantiles from | 2 / (1 - level) - 1 errors (19 for 90%) | order statistics: with n errors the next falls outside their range 2 / (n + 1) of the time |
| small-sample interval | mean ± t(n-1) · s · sqrt(1 + 1/n) | the textbook prediction interval for one new observation |
| `MIN_FIT`, `MIN_ERRORS` | 3 past values, 5 backtest errors | the least a mean or a spread can be computed from with any meaning |
| releases ahead of a revision target (`forecast.revision`) | the days to the resolving release over the mean gap between the series' past releases, rounded, at least 1 | counting: a series with two releases a month (advance and full report) has a median gap of about three weeks, which counted one release where there are two |
| an annual or comprehensive update (`forecast.BENCHMARK`) | a release whose median revision across the reference months is over 10 times the series' usual mean revision per release; revisions across it are left out of the history | a whole history moved at once is not the revision the next release makes; 10 is an order of magnitude, far from both normal releases (about 1x) and the one update in the public units (over 100x) |
| a prior joined to the model's answer (`priors.combine`) | the point is the model's, unless the prior decides it (below) or the model gave none; the interval is the prior's band around the point | an ablation on the 30 runs of image c3-4195943: the model's point did as well as the 0.5 average of the two (+0.002 analysis) and better than the prior's (+0.006); no weight is fitted |
| priors that decide the point (`Forecast.decides`) | a growth from a copied level, and a revision (its label the point's side of the latest estimate); shown to the model as its entity's decided forecast before it writes, so the reasons explain the submitted number | growth: moved back one quarter on the filings' own figures (pre-cutoff), the level carried forward beat the latest growth rate carried forward (the model's habit), MAE 25.4 against 40.6; revision: the model called directions at chance in every version |
| ranking targets | the model's point kept; the interval centred on it with the prior's width | priors chosen entity by entity are not comparable across entities (the average reordered a ranking); the prior's offsets carry its own method's past drift, not the model's |
| steps to a resolution date (`forecast.steps_between`) | whole periods of the series' cadence that end on or before it | an observation after the resolution date is not in the target |
| credit-event labels from the filings (`signals`), shown to the model with the deciding passage | substantial doubt about the issuer's ability to continue as a going concern (not negated or merely evaluated) means credit_event; a disclosed investment-grade rating with no speculative grade and no such doubt means no_event | the first solvency signal the task names; investment-grade one-year default rates are a fraction of a percent. Checked on the public unit's 8 filings only |
| least half-width of a yield change's interval (`moves.A`, `moves.K`) | hypot(28, 1.05 x the largest move the documents state, scaled to the horizon by the square root of weekdays), the earlier levels copied with their dates (`extract.EARLIER_LEVEL`) | A and K fitted on the FOMC validation set (58 FOMC meetings built from public FRED data in the same format as the practice units, all resolved by 2022-06-14, before both public cutoffs; not included in this repository) |
| horizons the yield floor's A holds for (`moves.HORIZONS`) | 27 to 63 weekdays; outside them A scales with the square root of the horizon over the nearer end | the horizons of the 58 validation meetings A was fitted on (shortest 27, longest 63 weekdays); a constant is not carried to a horizon it was never fitted on |
| a copied level's distance from its base (`extract.RATIO`) | a copied level must lie between a fifth and five times its base, or it is left out (and a copied table row's first column is taken only within that range) | a value that far off is another quantity, period or scale (net income for EPS, thousands for units); quarterly EPS growth inside the range spans -80% to +400% |
| House calls before the client stops (`llm.BREAK_AFTER`) | 3 calls in a row that could not reach the House (connection or timeout); a rate limit or server error never counts | operational, not a scoring choice |
| waits before retrying a rate limit or server error (`llm.THROTTLED`) | 5, 15, 30, 60 s, or the reply's Retry-After when longer, within the clock | operational |
| prompt tokens in flight at once (`pipeline.TOKENS_IN_FLIGHT`, `IN_FLIGHT`) | 200,000 and at most 4 calls; a call starts as soon as one ends (`pipeline.admits`), a single larger call alone | operational: keeps units of large filings from meeting a token-rate limit all at once, without waiting for a wave's slowest call |
| samples of a unit's answer calls (`pipeline.SAMPLES`, `SAMPLE_CALLS`) | the most of 5 or 3 whose calls fit in 15 (else 1), and 1 when every entity's point is its prior's (model weight 0); merged per entity: majority label, the median sample's point and claims, an interval spanning those samples' intervals | odd so a label vote has no tie; 15 leaves room in the 25 requests for extraction. 3 adopted after a Development comparison of combined stored runs (spread halved); 5 where they fit after a bootstrap of the stored samples (same means, narrower spread); the spanning interval replaced mean bounds, which an outlying sample dragged off the median point (no label is used at run time) |
| corrective turns and asking again | none for answer calls (extraction calls keep one corrective turn) | the ablation on the same 30 runs: neither changed a submitted number or a false-claim count; an entity no sample answers takes its prior, else the floor |
| growth prior from a copied level (`solution/extract.py`, `priors.growth`) | for a growth target with no history series and a `prior_*` base in its row: the latest level the entity's documents report, copied by the House model with an exact quote (one corrective re-ask), kept only when the quote is in its own documents, the value is in the quote and differs from the base; combined at the same weight 0.5, the model's band stretched to hold both | the rule and its checks are generic; building it followed the Development diagnosis that the model carried a quarter's growth rate to the next |

## Models

The House model only, as `submission.json` `models[]` declares it (`nvidia/nemotron-3-super-120b-a12b`, revision
`rl-030326-fp8`, `access: "api"`). No local model, adapter or neural component is in the image.

## Data and cutoffs

- Training, fitting, model selection, calibration: none. The image contains no fitted parameter and no data beyond
  the static assets above.
- Evaluation inputs and citations: only `/input` (the task and its frozen corpus). Documents dated after the cutoff
  are dropped on load. Nothing is fetched; the only network call is the House model.
- Constants were set from principle, document structure and reply lengths (reply budgets from reply token counts),
  or fitted on pre-cutoff data (the yield floor's A and K on the FOMC validation set), as the table above records.
- During Development, prompt and pipeline versions were compared on the public practice units against locally
  reconstructed outcomes, as the Development leaderboard compares them. Those outcomes never enter the image. See
  "Practice-unit outcomes".

## Practice-unit outcomes (Development only; issue #24 ruling)

The organizers ruled on issue #24 (2026-10-05 and 2026-10-07, for every team) that the ten practice units' realized
outcomes may inform constants and design choices for the Final submission, for every Final task, provided the values
are the first-published ones, each with its source and retrieval date recorded here, and the practice units are named
as the in-sample exception.

**In-sample exception.** The ten practice units below, and our Development scores on them, are in-sample: their outcomes
informed our choices. Those scores are not a measure of out-of-sample skill.

**What the outcomes informed.** Local scoring against them chose between whole designs and components: the comparison
of prompt and output-format versions (lab v4.x); the sample merge (the median sample with a spanning interval) and the
number of samples; the two-rule combination of the model's point with the computed priors and which priors decide the
point (revisions, copied growth levels); the credit-event filing signals; the call batching, the extraction checks; and
the ablations that removed corrective turns, asking again, the level guard and the document segmentation stack
([results.md](results.md#ablations)). No numeric constant was fitted or tuned on them: the
constants in "Run-time statistics" are set by principle, by counting, or on pre-cutoff data.

**The outcomes used.** Reconstructed locally and kept outside the image and outside this repository (never under
`units/`) by a set of scripts (`fetch.sh` downloads, `alfred.py` and `eps.py` read the vintage and
filing data, `build_keys.py` writes the keys), retrieved on 2026-09-28 (the downloads' timestamps). Before writing a
key, `build_keys.py` asserts that its source reproduces the task's own pre-cutoff values (the fomc start yields, the
cotpos net position and open interest on 2024-10-22, the prior-year EPS, the pre-earnings closes, CPI's September
change, the macrorev pre-cutoff estimates), so each source is the series the task was built from.

| practice unit | outcome | source | first-published status |
|---|---|---|---|
| t4-fomc-curve-20220728 | constant-maturity yield change, 2022-07-28 to 2022-09-20 | FRED DGS2/3/5/7/10/30 | Treasury par yields are not revised |
| t4-fomc-curve-20240918 | constant-maturity yield change, 2024-09-19 to 2024-11-06 | FRED DGS2/3/5/7/10/30 | Treasury par yields are not revised |
| t4-auction-btc-202411-us7 | bid-to-cover ratio of each November 2024 auction | Treasury Fiscal Data auctions_query (official, excludes SOMA) | official auction results, not revised |
| t4-cpicomp-202410-us11 | October 2024 month-over-month change of each CPI component | ALFRED vintage 2024-11-13 (SA index levels) | the first print, by vintage |
| t4-macrorev-20240930-us6 | each series' next published estimate and its direction | ALFRED vintage on each row's resolving release date | the first publication of that estimate, by vintage |
| t4-cotpos-202411-us10 | change in noncommercial net position, % of open interest, 2024-10-22 to 2024-11-26 | CFTC Legacy Futures-Only COT (publicreporting.cftc.gov, dataset 6dca-aqww) | the dataset as retrieved; it is not vintage-stamped, and later CFTC corrections, rare, could be included |
| t4-eps-yoy-2023Q2-mixed | Q2 2023 GAAP diluted EPS | SEC XBRL companyconcept us-gaap:EarningsPerShareDiluted (data.sec.gov) | the earliest-filed XBRL fact for the quarter, i.e. the 10-Q that first reported it (accession and filing date recorded per company) |
| t4-eps-growth-2024Q3-banks | Q3 2024 GAAP diluted EPS growth against the prior-year quarter | SEC XBRL companyconcept us-gaap:EarningsPerShareDiluted (data.sec.gov) | the earliest-filed XBRL fact for the quarter, i.e. the 10-Q that first reported it (accession and filing date recorded per bank) |
| t4-postearn-20240201-megacap | one-day abnormal return after 2024-02-01 earnings, against SPY | Yahoo Finance daily closes | closes; no dividend in the window, and none of AAPL, AMZN, META or SPY split after it, so the split-adjusted closes equal the published ones |
| t4-credit-event-2023 | credit event within 12 months of 2023-03-31 | company press releases and bankruptcy dockets (Kroll, Epiq), entered by hand in `build_keys.py` | event dates (petition filings), not revised values |

A key was also reconstructed for the shipped exemplar `t4-EXAMPLE-eps-beat` (its Q1 2024 GAAP diluted EPS, the
earliest-filed SEC XBRL fact for the quarter), retrieved on 2026-09-28; it is in-sample in the same way.
