# Statistical Significance of the Selection Ranker

Generated at: 2026-08-16 12:09:22

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

### gbm

| Statistic | Real | Null mean | p-value | Verdict |
|---|---:|---:|---:|---|
| Mean AUC | 0.5022 | 0.5000 | 0.1741 | not significant (indistinguishable from chance) |
| Precision@N | 0.5333 | 0.4981 | 0.0448 | significant at 5% (evidence of real skill) |

- **Base rate** (label = 1 frequency): 0.4994
- **Bootstrap 95% CI on per-fold mean AUC**: [0.4954, 0.5092] (median 0.5022, 132 folds)

Null distribution of mean AUC over 200 label shuffles:

```
  0.4947–0.4953 | ###
  0.4953–0.4959 | #######
  0.4959–0.4965 | ################
  0.4965–0.4971 | ##############
  0.4971–0.4976 | ##########
  0.4976–0.4982 | ############################
  0.4982–0.4988 | #####################################
  0.4988–0.4994 | ##############################
  0.4994–0.5000 | ######################################
  0.5000–0.5005 | #################
  0.5005–0.5011 | ###################################
  0.5011–0.5017 | ########################################
  0.5017–0.5023 | ###################  <- REAL
  0.5023–0.5028 | #################
  0.5028–0.5034 | ############
  0.5034–0.5040 | ##############
  0.5040–0.5046 | #######
  0.5046–0.5052 | ##
  0.5052–0.5057 | 
  0.5057–0.5063 | ##
```

### logistic

| Statistic | Real | Null mean | p-value | Verdict |
|---|---:|---:|---:|---|
| Mean AUC | 0.5012 | 0.4998 | 0.2935 | not significant (indistinguishable from chance) |
| Precision@N | 0.5364 | 0.4997 | 0.0348 | significant at 5% (evidence of real skill) |

- **Base rate** (label = 1 frequency): 0.4994
- **Bootstrap 95% CI on per-fold mean AUC**: [0.4913, 0.5109] (median 0.5011, 132 folds)

Null distribution of mean AUC over 200 label shuffles:

```
  0.4941–0.4947 | #####
  0.4947–0.4952 | ##########
  0.4952–0.4958 | #######
  0.4958–0.4963 | ##############
  0.4963–0.4969 | #########
  0.4969–0.4974 | ###################
  0.4974–0.4979 | ##############
  0.4979–0.4985 | ##########
  0.4985–0.4990 | ########################
  0.4990–0.4996 | ############################
  0.4996–0.5001 | #####################################
  0.5001–0.5007 | ##############################
  0.5007–0.5012 | ########################################  <- REAL
  0.5012–0.5017 | ##########################
  0.5017–0.5023 | ##########################
  0.5023–0.5028 | #####################
  0.5028–0.5034 | ############
  0.5034–0.5039 | #####
  0.5039–0.5045 | #####
  0.5045–0.5050 | #####
```


## Interpretation

A low p-value means random outcomes rarely reproduce the observed ranking quality,
so the edge is unlikely to be luck. A precision@N that is significant even when the
whole-cross-section AUC is only marginal tells you the skill concentrates in the
**top of the ranking** — exactly where the strategy acts. This is an out-of-sample
research evaluation and does not guarantee future performance.
