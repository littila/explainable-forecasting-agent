# Walkthroughs

Five practice units followed through one stored run (build c4, run r1), each with a different split of work between
code and model. Numbers come from the run's request trace and answer, from replaying the code offline (the replay
reproduces every submitted point and interval), and from the units' outcomes reconstructed from public data. Since
build c5 the prompts also open with the task's facts (target, label definitions, dates), and the Final build's yield
floor has no fitted constant (the yield walkthrough shows the Final rule); the rest is unchanged.

## EPS growth of eight banks: a copied level decides

*Regression: year-over-year growth of Q3 2024 diluted EPS, in percent. Cutoff 2024-10-10; 8 banks; 12 filings; 16
requests; 209 s.*

1. **Fitting the filings.** Every 10-Q is larger than one call's document budget (96,665 tokens). JPM's 10-Q (215,796
   tokens) is cut into windows of about 16,000 characters, ranked by BM25 against the task, and the best 50.4% is
   shown with gap markers.
2. **Copying the latest EPS.** The filings hold no dated series, so code asks the model to copy each bank's latest
   reported EPS with a quote, and checks it. The checks caught three different mistakes: JPM's copy of the base
   quarter itself (the first-column rule took 6.12 from the quoted row), a remembered figure in another run (USB 1.03,
   its real Q3 EPS, published after the cutoff and in no document; rejected because its period ends after the
   documents' date), and a value missing from its own quote.
3. **Deciding the point.** Growth = (level - base) / |base|: BAC 0.83 / 0.90 gives -7.78%, JPM 6.12 / 4.33 gives
   +41.34%. The figure goes on the bank's line as a decided forecast, and the model writes the reasons for it.

| Bank | Truth % | Point | Interval | Covered |
|---|---|---|---|---|
| BAC | -10.00 | -7.78 | [-15.0, 0.0] | yes |
| C | -7.36 | -6.75 | [-15.0, 1.5] | yes |
| GS | 53.56 | 57.59 | [40.0, 75.0] | yes |
| JPM | 0.92 | 41.34 | [31.34, 51.34] | no |
| MS | 36.23 | 31.88 | [25.0, 40.0] | yes |

**What it shows.** Data decides and the model reads and copies: carrying the last quarter's growth forward (the
model's own habit) scores MAE 17.3, the copied level 8.4. The weak spot is what one level cannot see: JPM's Q2 EPS
includes a $7.9 billion Visa gain that its filing flags, so its point misses by 40 points.

## Macro revisions: the revision prior decides

*Classification: will the next published estimate of a reference month be revised up or down? 12 series-months (payrolls,
industrial production, durable goods, housing starts, personal income, retail sales); 6 ALFRED vintage tables; 2
requests; 17 s.*

Code reads each series' vintage table and adds to the latest estimate the median of past revisions over the same number
of releases at the same age (payrolls July: -57, -2, -61, so down, 158,580). A release that moved the whole history at
once (personal income, +700 on every month) is left out. Every entity's forecast is decided before the call, so one
sample is enough and all three runs are identical. The model invented three revision sentences for a month not yet
revised; they were found in no document and were not cited.

**What it shows.** Without the prior, accuracy falls to 0.444 and the interval score more than doubles; its bands
covered 7 of 12 against the model's 2. The limit is the forecast itself: the truth was 6 up and 6 down, and the prior
said down 11 times.

## Credit events: the filings decide half the roster

*Classification: a bankruptcy, payment default or distressed exchange within 12 months of 2023-03-31? 8 issuers, one
filing each (0.18 to 1.03 MB); 8 requests; 101 s.*

Two plain rules read each issuer's full filing: a going-concern doubt decides credit_event (Bed Bath & Beyond,
WeWork), an investment-grade rating with no speculative grade decides no_event (Best Buy, Walgreens). Macy's split
ratings veto the second rule. The model decides the other four and misses one in every run: Yellow's 10-K has neither
signal, and the model read a $21.8 million profit and $241.8 million of liquidity as safety. Yellow filed for Chapter 11
on 2023-08-06. Accuracy 7 of 8 in every run.

## Treasury yields: the model's point, the data's interval

*Regression: the change in six constant-maturity yields, in basis points, from 2024-09-19 to 2024-11-06 (34 weekdays).
3 shared documents; 7 requests; 26 s.*

The rates snapshot holds 11 daily closes, too few to backtest a 34-day move, so the prior is "no change" with the daily
volatility scaled by the square root of the horizon. The model sets the point from the documents (the 50 bp cut, the
projections against market pricing), sampled five times: 15, 45, 15, 15, -5 for the 2-year. A copy call finds the one
older yield a sentence states (the 2-year at 4.77% on 2024-07-01, a 118 bp move over 58 weekdays), which sets a
floor of `118 x sqrt(34/58) = 90.4` bps on every interval.

| Interval | Mean interval score | Covered |
|---|---|---|
| Model's spanning intervals | 827 | 0 of 6 |
| Prior band around the model's point | 489 | 0 of 6 |
| With the yield floor (submitted) | 181 | 6 of 6 |

**What it shows.** The model reads the direction right (yields rose) but sizes the move at +2 to +15 bps against a
realised +54 to +78; three short documents cannot tell the size of a seven-week move. The data's contribution is the
interval.

## Futures positioning: the model ranks

*Ranking: ten futures markets ordered by the change in speculative net positioning, as a percent of open interest,
between two weekly reports. 10 own tables and 2 shared documents; 2 calls x 5 samples; 60 s.*

Code finds each market's series by the target's words (`net_%OI` reads as net, pct, oi) and backtests no change against
drift over 26 weekly rows: nine markets choose no change. The model makes the ranking from positioning crowdedness and
the macro backdrop, and gets the broad order right (Treasuries and natural gas near the top, gold and silver near the
bottom; Spearman 0.59). The backtested width placed around the model's point gave the best interval of three
(interval score 28.8, against 34.9 for the model's own and 44.7 for the prior's).

**What it shows.** The model's points are compressed (-6 to +6 against a truth from -17 to +7), so the two biggest
movers sit near zero, and each call ranks only its five markets while the scorer ranks all ten.
