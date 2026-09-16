# Statistical Significance of the Selection Ranker

Generated at: 2026-08-15 21:34:55

## Method

The walk-forward selector produces one out-of-sample score per (rebalance date,
ticker). Holding those **real scores fixed**, the outcome labels are shuffled
**within each rebalance date's cross-section** and the ranking statistic (mean
AUC, precision@N) is recomputed — 100 times — to build the "no-skill" null
distribution. The p-value is `(1 + #{null >= real}) / (1 + 100)`. Because
only labels are permuted (never re-training, never crossing dates), the test is
leakage-free and makes no distributional assumptions. Bootstrap resampling of the
per-fold AUCs adds a 95% confidence interval — the error bar a single backtest
lacks.

## Configuration

- Universe: 503 tickers
- Top-N: 3 · Feature normalisation: cross-sectional rank (leakage-safe)
- Label: forward 1-month return vs cross-sectional median
- Permutations: 100

## Results

### gbm

| Statistic | Real | Null mean | p-value | Verdict |
|---|---:|---:|---:|---|
| Mean AUC | 0.4953 | 0.5000 | 0.9604 | not significant (indistinguishable from chance) |
| Precision@N | 0.4848 | 0.4997 | 0.7624 | not significant (indistinguishable from chance) |

- **Base rate** (label = 1 frequency): 0.4994
- **Bootstrap 95% CI on per-fold mean AUC**: [0.4869, 0.5047] (median 0.4954, 132 folds)

Null distribution of mean AUC over 100 label shuffles:

```
  0.4938–0.4944 | ########
  0.4944–0.4951 | ##
  0.4951–0.4957 | ##  <- REAL
  0.4957–0.4963 | ########
  0.4963–0.4969 | ##########
  0.4969–0.4975 | ############
  0.4975–0.4981 | ############
  0.4981–0.4987 | ##########
  0.4987–0.4994 | ############
  0.4994–0.5000 | ########################################
  0.5000–0.5006 | ##################
  0.5006–0.5012 | #########################
  0.5012–0.5018 | ##############################
  0.5018–0.5024 | ##############################
  0.5024–0.5031 | ########
  0.5031–0.5037 | ############
  0.5037–0.5043 | ##
  0.5043–0.5049 | ##
  0.5049–0.5055 | 
  0.5055–0.5061 | #####
```

### logistic

| Statistic | Real | Null mean | p-value | Verdict |
|---|---:|---:|---:|---|
| Mean AUC | 0.4997 | 0.5001 | 0.5743 | not significant (indistinguishable from chance) |
| Precision@N | 0.5253 | 0.4954 | 0.1485 | not significant (indistinguishable from chance) |

- **Base rate** (label = 1 frequency): 0.4994
- **Bootstrap 95% CI on per-fold mean AUC**: [0.4852, 0.5139] (median 0.4997, 132 folds)

Null distribution of mean AUC over 100 label shuffles:

```
  0.4946–0.4951 | ######
  0.4951–0.4956 | 
  0.4956–0.4962 | #########
  0.4962–0.4967 | ###############
  0.4967–0.4973 | #########
  0.4973–0.4978 | ############
  0.4978–0.4984 | #########
  0.4984–0.4989 | ########################################
  0.4989–0.4994 | ######################
  0.4994–0.5000 | ######################  <- REAL
  0.5000–0.5005 | #########################
  0.5005–0.5011 | ###############################
  0.5011–0.5016 | ############################
  0.5016–0.5021 | #####################################
  0.5021–0.5027 | ######
  0.5027–0.5032 | ######
  0.5032–0.5038 | #########
  0.5038–0.5043 | #########
  0.5043–0.5049 | ###
  0.5049–0.5054 | #########
```


## Interpretation

A low p-value means random outcomes rarely reproduce the observed ranking quality,
so the edge is unlikely to be luck. A precision@N that is significant even when the
whole-cross-section AUC is only marginal tells you the skill concentrates in the
**top of the ranking** — exactly where the strategy acts. This is an out-of-sample
research evaluation and does not guarantee future performance.
