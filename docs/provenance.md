# Data and artifact provenance

The competition requires every submission to document what its image is built from and how it respects each
task's cutoff (the Track 4 repository's `docs/TRAINING-POLICY.md` and `docs/ARTIFACT-POLICY.md`). This is that record,
as filed with the Final submission; paths refer to this repository.

## Summary

This record describes what the Track 4 submission image is built from, as the training and artifact
policies require. The image holds no model weights and no training data. Every forecast comes from the
House model through the API, except where a rule computed at run time from the unit's own pre-cutoff documents
decides it (a revision prior, a copied growth level, a filing's going-concern or rating disclosure), and those are
shown to the model before it answers. At evaluation time the image reads only the supplied task and its frozen corpus;
the BM25 index and the statistics are computed per run over that corpus and never stored. One set of numbers in the
image was fitted, the fallback token estimator, on text alone (no outcome). No constant was fitted, chosen or calibrated
on outcomes other than the ten practice units', which the organizers' ruling on issue #24 permits for every Final task
with the disclosure in "Practice-unit outcomes" below (the practice units are named there as the in-sample exception).
The yield-interval constants fitted on a separate 2015-2022 FOMC set were removed for the Final after the 9-10 October
ruling; that set is recorded below as diagnostic only. The submitted copy is documentation for organizer verification, kept
with the build source; it is not a descriptor field and is not added to the upload ZIP.

## Image

| item | value |
|---|---|
| Final image | `ghcr.io/littila/t4-solution:final-cabd97f` (commit cabd97f), linux/amd64 manifest `sha256:adad673ddb8503ecbdf94c666052ed4f12c254a75190d164a199f96d2de46ed4` (`submission.json`) |
| built from | this repository at that commit, `Dockerfile` |
| base image | `python:3.13-slim`, pinned by index digest `sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f` (Docker Hub) |
| Python packages | `solution/requirements.txt`: openai 3.16.2, pydantic 2.13.5, jsonschema 4.26.0, tokenizers 0.23.2 (exact pins); the dependencies pip resolved for them at build, as `pip freeze` lists them in the image: Python 3.13.16; annotated-types 0.8.0, anyio 4.15.1, attrs 26.1.0, certifi 2026.7.22, click 8.5.0, filelock 4.0.12, fsspec 2026.9.0, h11 0.16.0, hf-xet 1.6.0, httpcore 1.0.9, httpcore2 2.13.1, httpx 0.28.1, httpx2 2.13.1, huggingface_hub 1.33.0, idna 3.20, jiter 0.17.0, jsonschema-specifications 2025.9.1, packaging 26.3, pydantic_core 2.46.5, PyYAML 6.0.3, referencing 0.37.0, rpds-py 2026.9.1, sniffio 1.3.1, tqdm 4.70.1, truststore 0.10.4, typing-inspection 0.4.4, typing_extensions 4.16.0 |

## Components

| component | kind | source, revision, checksum | license |
|---|---|---|---|
| `solution/` | our code: rendering, planning, prompts, parsing, citation resolution, assembly | this repository | repository `LICENSE` (MIT) |
| `solution/tokenizer/tokenizer.json` | static asset: tokenizer vocabulary only, no weights; counts prompt tokens | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` at revision `744b1880a37996c5d56bf454ae164dfd74d77c4e`, sha256 `623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7` (checked by the `Dockerfile`) | NVIDIA Nemotron Open Model License (`solution/tokenizer/LICENSE`, `NOTICE`) |
| fallback token estimator (`solution/tokens.py` `_FIT`, `_FIT_FLOOR`) | a linear fit of token counts on digits, words and other characters (1.7126, -0.4207, 0.2576), scaled by 1 / 0.8 so it undercounts in fewer than 5% of sections; used only when the tokenizer file cannot be read | fitted on the exact token counts of 776 rendered sections of the public units' corpora: text only, no outcome | repository `LICENSE` |
| `solution/analysis.schema.json` | static asset: answer schema | vendored from the toolkit `qfbench2-common` v2.5.1; a test asserts it equals the installed copy | toolkit license |
| `baselines/guardrails_example/__init__.py`, `citation_rail.py` | organizers' code: the reason checks (`check_submitted_reasons`, `reasons_judged`), standard library only on that path | this repository (organizer baseline, as merged from upstream at scorer 5.3.0) | repository `LICENSE` |
| BM25 index (`solution/retrieve.py`) | built at run time over the unit's own frozen corpus to fit long filings; never stored | - | - |

## Run-time statistics (`solution/data.py`, `solution/forecast.py`, `solution/priors.py`)

For an entity whose own history is a dated series in its unit's corpus, each run computes a prior from that history
alone. The corpus holds only documents dated on or before the cutoff (the embargo is applied at load), and every row of
those documents is read as written. A fixed menu of standard methods (last value, mean of the last 3 or 6, mean or
median of all; for a change, no change or drift; for a vintage table, the median past revision at the same age) is
backtested on the series' own history in those documents and the smallest backtest error picks the point. The interval
comes from that method's backtest errors; below MIN_ERRORS errors it is the last value (or no change) plus or minus
t(n-1) times the step spread times the square root of the steps. Nothing is fitted offline, stored, or chosen on
outcomes; the model's point is combined with the prior at run time.

| constant | value | how it was set |
|---|---|---|
| interval quantiles | (1 - level) / 2 and its complement | the interval's definition (5% and 95% for 90%) |
| empirical quantiles from | 2 / (1 - level) - 1 errors (19 for 90%) | order statistics: with n errors the next falls outside their range 2 / (n + 1) of the time |
| small-sample interval | mean ± t(n-1) · s · sqrt(1 + 1/n) | the textbook prediction interval for one new observation |
| `MIN_FIT`, `MIN_ERRORS` | 3 past values, 5 backtest errors | the least a mean or a spread can be computed from with any meaning |
| releases ahead of a revision target (`forecast.revision`) | the days to the resolving release over the mean gap between the series' past releases, rounded, at least 1 | counting: a series with two releases a month (advance and full report) has a median gap of about three weeks, which counted one release where there are two |
| an annual or comprehensive update (`forecast.BENCHMARK`) | a release whose median revision across the reference months is over 10 times the series' usual mean revision per release; revisions across it are left out of the history | a whole history moved at once is not the revision the next release makes; 10 is an order of magnitude, far from both normal releases (about 1x) and the one update in the public units (over 100x) |
| a prior joined to the model's answer (`priors.combine`) | the point is the model's, unless the prior decides it (below) or the model gave none; the interval is the prior's band moved to the point and widened to still cover the prior's own band | an ablation on the 30 runs of image c3-4195943 (practice units): the model's point did as well as the 0.5 average of the two (+0.002 analysis) and better than the prior's (+0.006); no weight is fitted |
| a prior of another scale (`priors.SCALE`) | a prior that does not decide is left out when the model's point lies more than 10 of the prior's band widths from the prior's point (a level for a change, percent for basis points) | an order of magnitude past a 90% band cannot be the same quantity on the same scale. A first form (band widths more than 10x apart) was dropped because a replay of the practice runs showed it firing on a same-scale entity whose model band was merely narrow |
| priors that decide the point (`Forecast.decides`) | a growth from a copied level, and a revision (its label the point's side of the latest estimate); shown to the model as its entity's decided forecast before it writes, so the reasons explain the submitted number | growth: moved back one quarter on the practice filings' own figures (pre-cutoff), the level carried forward beat the latest growth rate carried forward (the model's habit), MAE 25.4 against 40.6; revision: the model called directions at chance in every version (practice units) |
| a revision prior on a change target | none: a revision forecasts the estimate's next level, so a target named as a change takes no prior from it | the quantity's definition |
| ranking targets | the model's point kept; the interval centred on it with the prior's width | priors chosen entity by entity are not comparable across entities (the average reordered a ranking); the prior's offsets carry its own method's past drift, not the model's |
| steps to a resolution date (`forecast.steps_between`) | weekdays for a daily series (median gap of 3 days or less), else whole periods of the series' cadence that end on or before the date | an observation after the resolution date is not in the target |
| credit-event labels from the filings (`signals`), shown to the model with the deciding passage | substantial doubt about the issuer's ability to continue as a going concern (not negated or merely evaluated) means credit_event; a disclosed investment-grade rating with no speculative grade and no such doubt means no_event | the first solvency signal the task names; investment-grade one-year default rates are a fraction of a percent. Checked on the practice unit's 8 filings only |
| least half-width of a yield change's interval (`moves.half_width`) | the largest move the unit's documents state between an earlier level and the cutoff level, scaled to the horizon by the square root of weekdays; none when no earlier level is stated. A yield change is a target whose name holds "yield" and a basis-point word (`types.BPS`) | a random walk's spread grows with the square root of time; no constant. Replaces A and K (see "Diagnostic only"). On the two practice fomc units the analysis estimate is unchanged (the interval leg is capped by the prediction leg there) |
| a copied level's distance from its base (`extract.RATIO`) | a copied level must lie between a fifth and five times its base, or it is left out (and a copied table row's first column is taken only within that range) | a value that far off is another quantity, period or scale (net income for EPS, thousands for units); quarterly EPS growth inside the range spans -80% to +400% |
| a zero-width interval (`assemble.ZERO_WIDTH`) | an interval narrower than 1e-4 x max(1, the point's size) takes, around its point, the median width of the unit's entities with a positive one (the rule unanswered entities already follow) | a zero-width 90% interval never holds a continuous outcome; 1e-4 absorbs the rounding of bounds a reply echoes. Found in a replay of practice runs (a decided point echoed as both bounds) |
| floor rows (`assemble`) | an entity no reply answers: its prior, else no change or its row's latest value (`floor_point`), the median width of the answered entities, the most common label, else the vocabulary's neutral label (`NEUTRAL`: inline, in_line, unchanged, no_change, flat, neutral, hold, stable, none, no_event); a citation of 20 to 600 characters holding a figure (`FLOOR_MAX`) | the least-assumption answer; the neutral words are the ones label vocabularies use for "nothing happens" |

## Prompts, budgets and operation

| constant | value | how it was set |
|---|---|---|
| sampling (`llm.TEMPERATURE`, `TOP_P`, `SEED`) | 0.0, 0.95, the constant 20260731; thinking off | greedy for stable predictions (decided 2026-09-28; at 0 the exemplar's label held on 6 of 6 runs); the seed is a constant because the rerun's `QFBENCH_SEED` is fresh (SUBMISSION_CLI.md rule 4) |
| samples of a unit's answer calls (`pipeline.SAMPLES`, `SAMPLE_CALLS`) | the most of 5 or 3 whose calls fit in 15 (else 1), and 1 when every entity's point is decided by its prior; merged per entity: majority label, the median sample's point and claims, an interval spanning those samples' intervals | odd so a label vote has no tie; 15 leaves room in the 25 requests for extraction. 3 adopted after a Development comparison of combined stored runs (spread halved); 5 where they fit after a bootstrap of the stored samples (same means, narrower spread); the spanning interval replaced mean bounds, which an outlying sample dragged off the median point (practice units) |
| corrective turns and asking again | none for answer calls (extraction calls keep one corrective turn); one re-ask (`pipeline._again`) for an entity no sample answered and no prior covers, within the request budget; else the floor | the ablation on the 30 c3 runs: corrective turns changed no submitted number or false-claim count; the re-ask never fired on the practice units |
| growth prior from a copied level (`solution/extract.py`, `priors.growth`) | for a growth target with no history series and a `prior_*` base in its row: the latest level the entity's documents report, copied by the House model with an exact quote (one corrective re-ask), kept only when the quote is in its own documents, the value is in the quote and differs from the base; it decides the point, and the interval is the model's band moved to that point and stretched to hold the model's own point | the rule and its checks are generic; building it followed the Development diagnosis that the model carried a quarter's growth rate to the next |
| prompt and output budgets (`render.CONTEXT_SHARE` 0.4, `PROMPT_RESERVE` 8,192, `WINDOW` 262,144; `plan.OUTPUT_SHARE` 0.85, `BATCH_MAX` 16, `PER_REASON` 300, `PER_BLOCK` 160, `SHARED` 2, `TAG` 256; `extract.PER_ENTITY` 250) | a call's prompt within 40% of the context window; replies planned within 85% of the 4,000-token cap | reply budgets from reply token counts (lab v4.11's means of 242 and 135 tokens, plus about 20%); the window is the House context the organizers published; the planner is lab v4.12's, the citation resolver lab v3.16k's |
| a long document in a call (`render.fit`, `WINDOW_CHARS` 16,000, `MIN_WINDOW_CHARS` 200, `GAP_TOKENS` 16) | windows of 16,000 characters, or of half the document's cap in characters when smaller (never under 200), ranked by BM25 against the task and kept best-first while they fit | windows sized to the cap so a small cap still shows the best passage (with fixed windows, a cap under one window showed the document blank) |
| documents a call shows (`plan.MIN_DOC`) | at most the call's budget over 1,000 tokens; an entity with more shows its latest that many, the others staying citable; a unit no plan fits is asked with its entity rows alone (`plan.rows_only`) | about a page of text each, so hundreds of short documents are not each cut to nothing; the most recent carry the latest information |
| corpus files (`corpus.load_unit`) | with a manifest, only the files it declares are loaded; a null `entity_ids` reads as no label | the scorer resolves citations through the manifest, so citing an undeclared file would refuse the unit |
| request split and clock (`pipeline.COPY_CALLS` 8, `COPY_REQUESTS` 12, `EARLIER_REQUESTS` 2, `MIN_CALL` 30 s, `REASONS_AFTER` 10 s; `cli.HARD_DEADLINE` 570 s, `LIMIT_MARGIN` and `SOFT_MARGIN` 30 s) | copies take at most 12 of the 25 requests; the hard deadline is the card's time limit less 30 s, the soft one 30 s before it | operational: the platform's 25 requests and 600 s per unit |
| House calls before the client stops (`llm.BREAK_AFTER`) | 3 calls in a row that could not reach the House (connection or timeout); a rate limit or server error never counts | operational, not a scoring choice |
| waits before retrying a rate limit or server error (`llm.THROTTLED`, `RETRY_MIN_LEFT` 60 s, `CALL_TIMEOUT` 900 s) | 5, 15, 30, 60 s, or the reply's Retry-After when longer, within the clock | operational |
| prompt tokens in flight at once (`pipeline.TOKENS_IN_FLIGHT`, `IN_FLIGHT`) | 200,000 and at most 4 calls; a call starts as soon as one ends (`pipeline.admits`), a single larger call alone | operational: keeps units of large filings from meeting a token-rate limit all at once |
| reason selection (`reply.selected`) | at most 3 reasons the grader's checks pass, cited ones first, then by how much of the roster a reason covers | the reasons cap (3) and the grader's checks; the order is a design choice informed by the practice units |
| copied from the scorer and the platform | `FUNCTION_WORDS`, `META_WORDS`, `FILLER_ANCHORS`, `SPAN_MAX` 8,000, `cite.CLAIM_MAX` 600, `QUOTE_MIN` 40, `QUOTE_WORDS` 8, `REASONS_MAX` 3, `MAX_REQUESTS` 25, `MAX_OUTPUT` 4,000; BM25 k1 1.5, b 0.75; the regexes; `priors._in_percent` (a series within 0.5x to 2x of a percent-unit row value) | the scorer's and the platform's published rules, textbook defaults, and parsing |

## Models

The House model only, as `submission.json` `models[]` declares it (`nvidia/nemotron-3-super-120b-a12b`, revision
`rl-030326-fp8`, `access: "api"`). The image holds no model weights and no other neural component: the tokenizer file
is a vocabulary only.

## Data and cutoffs

- Fitting: the fallback token estimator only (`tokens._FIT`, text of the public units' corpora, no outcome).
- Selection and calibration: the ten practice units' first-published outcomes, under the issue #24 exception (below).
  No other outcome chose, kept or set anything in the image.
- Evaluation inputs and citations: only `/input` (the task and its frozen corpus). Documents dated after the cutoff or
  undated are dropped on load, and a manifest's undeclared files are not loaded. Nothing is fetched; the only network
  call is the House model.
- Constants were otherwise set from principle, document structure, the platform's limits and reply lengths, as the
  tables above record.

## Diagnostic only: the FOMC validation set

Built 2026-09-23 from FRED daily series DGS2/3/5/7/10/30 (US Treasury par yields via FRED, public data under the FRED
terms of use), one unit per regularly scheduled FOMC meeting from 2015-01-29 to 2022-05-05 (58 units; the cancelled
2020-03-17/18 meeting omitted), each labelled with the yield change in basis points from the day after the meeting to
the last trading day before the next one; the latest resolution date is 2022-06-14. Kept outside the image
and outside this repository.

What it was used for, in Development (September and October 2026): fitting the yield floor's constants A = 28 bps and
K = 1.05 (an 8-point grid by mean 90% interval score) and the range of horizons they held for (27 to 63 weekdays); and
comparing the model's point with no change on yield changes. Under the organizers' ruling of 9-10 October on issue #24,
outcomes that may postdate a Final task's cutoff cannot choose or calibrate anything for it, and the Final cutoffs are
sealed. So A, K and the horizon range were removed before the Final image, and the yield floor became the rule in the
table above, chosen on its rationale. The comparisons are kept here as diagnostic only: nothing in the Final image
depends on them. The model's point for yield changes stands on the practice-unit ablation recorded above.

## Practice-unit outcomes (issue #24 ruling)

The organizers ruled on issue #24 (2026-10-05 and 2026-10-07, for every team; restated 2026-10-09 and 2026-10-10) that
the ten practice units' realized outcomes may inform constants and design choices for the Final submission, for every
Final task, provided the values are the first-published ones, each with its source and retrieval date recorded here,
and the practice units are named as the in-sample exception.

**In-sample exception.** The ten practice units below, and our Development scores on them, are in-sample: their outcomes
informed our choices. Those scores are not a measure of out-of-sample skill.

**What the outcomes informed.** Local scoring against them chose between whole designs and components: the comparison
of prompt and output-format versions (lab v3.16k to v4.12); the sample merge (the median sample with a spanning
interval) and the number of samples (1, 3 or 5); the two-rule combination of the model's point with the computed priors
and which priors decide the point (revisions, copied growth levels); the credit-event filing signals; the call batching
and the extraction checks; the zero-width interval rule; the form of the prior-scale check; and the ablations that
removed corrective turns, the level guard and the document segmentation stack
([results.md](results.md#ablations)). No constant was fitted on them. The Final-round fixes of
2026-10-12 (document windows, document limits, loader, parsing) were checked by replaying 60 stored practice runs, whose
answers they leave unchanged apart from the yield intervals; that is a correctness check, not a score.

**The outcomes used.** Reconstructed locally and kept outside the image and outside this repository (never under
`units/`) by a set of scripts (`fetch.sh` downloads, `alfred.py` and `eps.py` read the vintage and
filing data, `build_keys.py` writes the keys), retrieved on 2026-09-28 (the downloads' timestamps). Before writing a
key, `build_keys.py` asserts that its source reproduces the task's own pre-cutoff values (the fomc start yields, the
cotpos net position and open interest on 2024-10-22, the prior-year EPS, the pre-earnings closes, CPI's September
change, the macrorev pre-cutoff estimates), so each source is the series the task was built from.

| practice unit | outcome | source | license | first available | first-published status |
|---|---|---|---|---|---|
| t4-fomc-curve-20220728 | constant-maturity yield change, 2022-07-28 to 2022-09-20 | FRED DGS2/3/5/7/10/30 | US Treasury data via FRED (FRED terms of use) | the 2022-09-20 closes | Treasury par yields are not revised |
| t4-fomc-curve-20240918 | constant-maturity yield change, 2024-09-19 to 2024-11-06 | FRED DGS2/3/5/7/10/30 | US Treasury data via FRED (FRED terms of use) | the 2024-11-06 closes | Treasury par yields are not revised |
| t4-auction-btc-202411-us7 | bid-to-cover ratio of each November 2024 auction | Treasury Fiscal Data auctions_query (official, excludes SOMA) | US government work, public domain | each auction's date, 2024-11-04 to 2024-11-27 | official auction results, not revised |
| t4-cpicomp-202410-us11 | October 2024 month-over-month change of each CPI component | ALFRED vintage 2024-11-13 (SA index levels) | BLS data via ALFRED (public domain; FRED terms of use) | 2024-11-13 | the first print, by vintage |
| t4-macrorev-20240930-us6 | each series' next published estimate and its direction | ALFRED vintage on each row's resolving release date | BLS and BEA data via ALFRED (public domain; FRED terms of use) | each row's resolving release, 2024-10 to 2024-11 (recorded per row in the key) | the first publication of that estimate, by vintage |
| t4-cotpos-202411-us10 | change in noncommercial net position, % of open interest, 2024-10-22 to 2024-11-26 | CFTC Legacy Futures-Only COT (publicreporting.cftc.gov, dataset 6dca-aqww) | US government work, public domain | the CFTC release of the 2024-11-26 report (2024-12-02) | the dataset as retrieved; it is not vintage-stamped, and later CFTC corrections, rare, could be included |
| t4-eps-yoy-2023Q2-mixed | Q2 2023 GAAP diluted EPS | SEC XBRL companyconcept us-gaap:EarningsPerShareDiluted (data.sec.gov) | SEC EDGAR data, public domain | each company's 10-Q filing date (accession and date recorded per company in the key) | the earliest-filed XBRL fact for the quarter, i.e. the 10-Q that first reported it |
| t4-eps-growth-2024Q3-banks | Q3 2024 GAAP diluted EPS growth against the prior-year quarter | SEC XBRL companyconcept us-gaap:EarningsPerShareDiluted (data.sec.gov) | SEC EDGAR data, public domain | each bank's 10-Q filing date (recorded per bank in the key) | the earliest-filed XBRL fact for the quarter, i.e. the 10-Q that first reported it |
| t4-postearn-20240201-megacap | one-day abnormal return after 2024-02-01 earnings, against SPY | Yahoo Finance daily closes | Yahoo Finance terms of service | the 2024-02-02 closes | closes; no dividend in the window, and none of AAPL, AMZN, META or SPY split after it, so the split-adjusted closes equal the published ones |
| t4-credit-event-2023 | credit event within 12 months of 2023-03-31 | company press releases and bankruptcy dockets (Kroll, Epiq), entered by hand in `build_keys.py` | public court records and company publications | each event's petition date (recorded per company in the key) | event dates (petition filings), not revised values |

A key was also reconstructed for the shipped exemplar `t4-EXAMPLE-eps-beat` (its Q1 2024 GAAP diluted EPS, the
earliest-filed SEC XBRL fact for the quarter), retrieved on 2026-09-28; it is in-sample in the same way.
