# Statistical Significance of the Selection Ranker

Generated at: 2026-09-18 18:04:19

## Method

The walk-forward selector produces one out-of-sample score per (rebalance date,
ticker). Holding those **real scores fixed**, the outcome labels are shuffled
**within each rebalance date's cross-section** and the ranking statistic (mean
AUC, precision@N) is recomputed — 200 times — to build the "no-skill" null
distribution. The p-value is `(1 + #{null >= real}) / (1 + 200)`. Because
only labels are permuted (never re-training, never crossing dates), the test is
leakage-free and makes no distributional assumptions. Bootstrap resampling of the
per-fold AUCs adds a 95% confidence interval — the error bar a single backtest
lacks.

## Configuration

- Universe: 503 tickers
- Top-N: 5 · Feature normalisation: cross-sectional rank (leakage-safe)
- Label: forward 1-month return vs cross-sectional median
- Permutations: 200

## Results

### logistic

| Statistic | Real | Null mean | p-value | Verdict |
|---|---:|---:|---:|---|
| Mean AUC | 0.4995 | 0.4998 | 0.5373 | not significant (indistinguishable from chance) |
| Precision@N | 0.5121 | 0.4826 | 0.0945 | marginal (weak evidence; not significant at 5%) |

- **Base rate** (label = 1 frequency): 0.4837
- **Bootstrap 95% CI on per-fold mean AUC**: [0.4867, 0.5131] (median 0.4996, 132 folds)

Null distribution of mean AUC over 200 label shuffles:

```
  0.4947–0.4953 | #############
  0.4953–0.4959 | #####
  0.4959–0.4965 | #############
  0.4965–0.4971 | ########
  0.4971–0.4977 | #####################
  0.4977–0.4983 | #############
  0.4983–0.4989 | #####################################
  0.4989–0.4995 | ########################################
  0.4995–0.5001 | ################################  <- REAL
  0.5001–0.5007 | ################################
  0.5007–0.5013 | ###########################
  0.5013–0.5019 | ################################
  0.5019–0.5025 | ###########
  0.5025–0.5031 | ######
  0.5031–0.5037 | ######
  0.5037–0.5043 | ###########
  0.5043–0.5049 | #####
  0.5049–0.5055 | ###
  0.5055–0.5061 | ##
  0.5061–0.5067 | ###
```


## Interpretation

A low p-value means random outcomes rarely reproduce the observed ranking quality,
so the edge is unlikely to be luck. A precision@N that is significant even when the
whole-cross-section AUC is only marginal tells you the skill concentrates in the
**top of the ranking** — exactly where the strategy acts. This is an out-of-sample
research evaluation and does not guarantee future performance.
