# Statistical Significance of the Selection Ranker

Generated at: 2026-09-18 11:30:18

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
| Mean AUC | 0.4934 | 0.4999 | 1.0000 | not significant (indistinguishable from chance) |
| Precision@N | 0.4750 | 0.4979 | 0.9005 | not significant (indistinguishable from chance) |

- **Base rate** (label = 1 frequency): 0.4994
- **Bootstrap 95% CI on per-fold mean AUC**: [0.4841, 0.5034] (median 0.4935, 132 folds)

Null distribution of mean AUC over 200 label shuffles:

```
  0.4945–0.4951 | ###
  0.4951–0.4957 | #######
  0.4957–0.4962 | #########
  0.4962–0.4968 | #########
  0.4968–0.4974 | #########
  0.4974–0.4980 | ###################
  0.4980–0.4986 | ###############################
  0.4986–0.4991 | ########################
  0.4991–0.4997 | #################################
  0.4997–0.5003 | ########################
  0.5003–0.5009 | ################
  0.5009–0.5015 | ########################################
  0.5015–0.5020 | #####################
  0.5020–0.5026 | #############
  0.5026–0.5032 | #############
  0.5032–0.5038 | #######
  0.5038–0.5044 | ######
  0.5044–0.5049 | #######
  0.5049–0.5055 | 
  0.5055–0.5061 | ####
```

### logistic

| Statistic | Real | Null mean | p-value | Verdict |
|---|---:|---:|---:|---|
| Mean AUC | 0.4991 | 0.4998 | 0.6119 | not significant (indistinguishable from chance) |
| Precision@N | 0.5076 | 0.4975 | 0.3881 | not significant (indistinguishable from chance) |

- **Base rate** (label = 1 frequency): 0.4994
- **Bootstrap 95% CI on per-fold mean AUC**: [0.4851, 0.5129] (median 0.4991, 132 folds)

Null distribution of mean AUC over 200 label shuffles:

```
  0.4938–0.4945 | ##########
  0.4945–0.4951 | ###
  0.4951–0.4958 | 
  0.4958–0.4965 | #####
  0.4965–0.4972 | ############################
  0.4972–0.4978 | ########################
  0.4978–0.4985 | ##############################
  0.4985–0.4992 | ######################################  <- REAL
  0.4992–0.4998 | ########################################
  0.4998–0.5005 | ########################################
  0.5005–0.5012 | #################################
  0.5012–0.5018 | #################################
  0.5018–0.5025 | ###################
  0.5025–0.5032 | ############
  0.5032–0.5038 | ################
  0.5038–0.5045 | #######
  0.5045–0.5052 | #####
  0.5052–0.5059 | 
  0.5059–0.5065 | ##
  0.5065–0.5072 | ##
```


## Interpretation

A low p-value means random outcomes rarely reproduce the observed ranking quality,
so the edge is unlikely to be luck. A precision@N that is significant even when the
whole-cross-section AUC is only marginal tells you the skill concentrates in the
**top of the ranking** — exactly where the strategy acts. This is an out-of-sample
research evaluation and does not guarantee future performance.
