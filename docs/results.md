# Results

Platform scores come from the organisers' Development runs on the ten practice units. Everything else was measured
locally: the agent run through its submission entrypoint on the same model, and scored against the practice units'
outcomes, which were reconstructed from first-published public data (see [provenance.md](provenance.md)). Raw metrics
(accuracy, Spearman, MAE, interval score) decided every change; the "local estimate" applies the scorer's formula with
stand-in baselines, since the official naive baselines are private. It matched the platform within 0.01 on 8 of 10
units for build c4.

## Development leaderboard

Analysis score (0 to 1) and leaderboard value (`-0.27 + 1.27 x analysis`), scorer 5.2.2 until 9 October, 5.3.0 after
(the board restarted at the switch, so 980107 is the entry that stood on it):

| Date | Submission | Build | Main change | Analysis | Leaderboard |
|---|---|---|---|---|---|
| 3 Oct | 958415 | v3.16k | first build scored by 5.2.2 | 0.4871 | +0.3486 |
| 5 Oct | 960771 | v4.12 | the 5.2.2 claim rules, reasons that carry the reasoning, grouped planner | 0.5553 | +0.4353 |
| 5 Oct | 960843 | v4.12 | a corrected reply keeps its missing entities; unanswered ones asked again | 0.6094 | +0.5039 |
| 5 Oct | 961834 | c | computed priors for entities with their own dated history | 0.5878 | +0.4765 |
| 5 Oct | 962251 | c2 | growth prior from copied levels, merged samples, House throttling | 0.5906 | +0.4801 |
| 7 Oct | 966184 | c3 | spanning merge, revision priors, filing signals, yield floor | **0.6634** | **+0.5725** |
| 7 Oct | 967146 | c4 | simplification: 44% less code | 0.6560 | +0.5631 |
| 12 Oct | 980107 | c6 | c5's generalization fixes (c5 itself was not uploaded), zero-width intervals widened, every row of admitted documents read | 0.6527 | +0.5590 |

966184 placed 6th on the Development leaderboard when it was scored. c3 to c4 removed 2,464 lines at the same score;
the 0.007 gap is one unit, a sampled ranking whose local runs spread from 0.755 to 0.878 Spearman.

## Ablations

Each post-call component switched off on the 30 stored runs of build c3 (replayed offline, no new model calls). A part
counts when it moves the mean by 0.005, or one unit by 0.02, in the same direction on most runs.

| Component | Mean | By unit | Raw metrics without it | Verdict |
|---|---|---|---|---|
| Computed priors | -0.037 | eps-growth -0.229, macrorev -0.132 | eps-growth MAE 8.4 to 21.9; macrorev accuracy 0.583 to 0.444 | keep |
| Growth copy | -0.023 | eps-growth -0.229 | MAE 8.4 to 21.9, interval score 95 to 265 | keep |
| Five samples (vs one) | -0.014 | eps-yoy -0.089, cotpos -0.040 | eps-yoy interval score 1.28 to 5.82 | keep |
| Filing signals | -0.011 | credit-event -0.111 | accuracy 0.833 to 0.750 | keep |
| Spanning interval | -0.008 | eps-yoy -0.071 | eps-yoy interval score 1.28 to 5.06 | keep |
| Yield floor | -0.004 | fomc-2022 -0.022 | interval score 328 to 1,166 | keep |
| 0.5 blend of points | +0.002 when replaced | fomc-2024 +0.016, cpicomp +0.013 | model's point at least as good | replaced |
| Corrective turns, asking again, level guard | 0.000 | none | never changed a number | removed |

## Per unit: c5 against c4

Local runs, three per unit, through the submission entrypoint on NVIDIA Nemotron 3 Super (OpenRouter route). c5 runs:
30 of 30 clean, 0 false claims, 0 failed calls, at most 24 requests and 98 seconds per unit.

| Unit | Type | c5 raw metrics (mean of 3) | c4 raw metrics | c5 est. | c4 est. | c4 platform |
|---|---|---|---|---|---|---|
| eps-yoy-2023Q2-mixed | classification | accuracy 1.00, interval score 1.08 | 1.00, 1.25 | 0.947 | 0.941 | 0.937 |
| cotpos-202411-us10 | ranking | Spearman 0.643, interval score 34.2 | 0.565, 27.4 | 0.720 | 0.680 | 0.758 |
| credit-event-2023 | classification | accuracy 0.875 | 0.875 | 0.750 | 0.750 | 0.750 |
| postearn-20240201-megacap | classification | accuracy 1.00, interval score 119 | 0.667, 128 | 0.849 | 0.727 | 0.729 |
| eps-growth-2024Q3-banks | regression | MAE 7.30, interval score 90 | 8.57, 103 | 0.676 | 0.641 | 0.625 |
| cpicomp-202410-us11 | regression | MAE 0.686, interval score 4.48 | 0.685, 4.31 | 0.609 | 0.611 | 0.608 |
| macrorev-20240930-us6 | classification | accuracy 0.583 | 0.583 | 0.583 | 0.583 | 0.583 |
| fomc-curve-20220728 | regression | MAE 76.1 bps, interval score 254 | 79.2, 302 | 0.558 | 0.548 | 0.542 |
| fomc-curve-20240918 | regression | MAE 61.6 bps, interval score 198 | 61.3, 198 | 0.532 | 0.532 | 0.530 |
| auction-btc-202411-us7 | regression | MAE 0.139, interval score 0.573 | 0.142, 0.646 | 0.504 | 0.500 | 0.496 |
| **mean** | | | | **0.673** | **0.651** | **0.656** |

Postearn's gain comes from stating each label's definition in the prompt: AAPL, labelled "flat" at 0 in every c4 run,
is labelled "negative_reaction" at -1.5 once the +-1% band is stated (true value -1.59%). One metric got worse: cotpos's
interval score, because silver's points moved toward zero against a true -15.

## Generalization fixes (c5)

An audit read every module for assumptions that hold on the practice units but may fail on the Final's held-out
units. Each fix has a test for the new shape; an offline replay of the 30 stored c4 runs confirmed that every fix
except the prompt change left the practice answers, citations and reasons identical.

| Could lose a unit | Fix |
|---|---|
| An entity's documents together exceed one call: the planner refused the unit | Its larger documents share the largest cap at which they fit |
| More calls than the 25-request cap | Copies take at most 12; the planner fits documents smaller to stay within the rest |
| Reason selection had no clock and read the whole corpus per check | Checked lazily in rank order, on the cited documents, with a deadline (6 min 12 s to 20 s over 30 replays) |
| An entity with nothing citable failed the schema | It cites its own task row |
| Loader crashes on shapes the scorer allows | Labels from definitions, text from spans, ids from file names, tolerant dates |

| Could degrade a unit | Fix |
|---|---|
| One unreadable series emptied every prior | Isolated per entity; monthly dates and one-release tables handled |
| `exchange_rate` matched "change"; growth from a rate field; basis points on a series already in bps | Whole-word triggers; level bases only; percent series only |
| A 28 bps floor on any yield horizon | Scaled outside the horizons it was fitted on; for the Final, replaced by a rule with no fitted constant |
| Unanswered rows got a zero-width interval and an arbitrary label | Asked once more when they have no prior; else the answered rows' width and label |
| The prompt never stated the label definitions or the resolution date | A task-facts block opens every task |

## Final

The Final entry, image `ghcr.io/littila/t4-solution:final-cabd97f` built from the code in this repository, ranked
**8th of 63** teams with a final score of **0.4441** (`-0.27 + 1.27 x analysis + 0.25 x reasoning`, on held-out units
mostly in families without a practice counterpart; the organisers publish no breakdown). The top three scored 0.4892,
0.4857 and 0.4768 ([leaderboard](https://www.agenthon.net/#leaderboard), published 3 November 2026).

What changed for the Final, and why, is in the [README](../README.md#final) and [provenance.md](provenance.md). Before
the upload the build was checked locally on the practice units, three runs through the submission entrypoint and one
through the image as the platform runs it (12 October 2026; 0 false claims, at most 21 of 25 requests and 144 seconds):

| Unit | Local estimate, 3 CLI runs | Image run | Raw metrics (3 runs) |
|---|---|---|---|
| auction-btc-202411-us7 | 0.507 | 0.521 | MAE 0.136, IS 0.535 |
| cotpos-202411-us10 | 0.698 | 0.693 | ranking quality 0.796, IS 31.6 |
| cpicomp-202410-us11 | 0.610 | 0.613 | MAE 0.684, IS 4.64 |
| credit-event-2023 | 0.750 | 0.750 | accuracy 0.875, IS 3.02 |
| eps-growth-2024Q3-banks | 0.652 | 0.656 | MAE 8.39, IS 92.5 |
| eps-yoy-2023Q2-mixed | 0.937 | 0.951 | accuracy 1.000, IS 1.4 |
| fomc-curve-20220728 | 0.555 | 0.561 | MAE 77.1, IS 395 |
| fomc-curve-20240918 | 0.536 | 0.532 | MAE 60.4, IS 181 |
| macrorev-20240930-us6 | 0.583 | 0.583 | accuracy 0.583, IS 1,963 |
| postearn-20240201-megacap | 0.732 | 0.731 | accuracy 0.667, IS 119 |
| **mean** | **0.656** | **0.659** | |

c6 scored 0.656 locally the same day; c5 had scored 0.673 four days earlier on replies c6 reproduces exactly, so the
difference between days is the model's, not the code's.
