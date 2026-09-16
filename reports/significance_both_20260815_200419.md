# Statistical Significance of the Selection Ranker

Generated at: 2026-08-15 20:04:19

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
| Mean AUC | 0.4958 | 0.5001 | 0.9505 | not significant (indistinguishable from chance) |
| Precision@N | 0.4975 | 0.5013 | 0.5545 | not significant (indistinguishable from chance) |

- **Base rate** (label = 1 frequency): 0.4994
- **Bootstrap 95% CI on per-fold mean AUC**: [0.4871, 0.5051] (median 0.4959, 132 folds)

Null distribution of mean AUC over 100 label shuffles:

```
  0.4946–0.4952 | #############
  0.4952–0.4958 | ###
  0.4958–0.4964 | #############  <- REAL
  0.4964–0.4970 | ##########
  0.4970–0.4976 | ####################
  0.4976–0.4982 | #################
  0.4982–0.4988 | #############
  0.4988–0.4994 | ####################
  0.4994–0.5000 | ########################################
  0.5000–0.5006 | #################################
  0.5006–0.5012 | ###########################
  0.5012–0.5018 | ########################################
  0.5018–0.5024 | ###########################
  0.5024–0.5030 | ###########################
  0.5030–0.5036 | ####################
  0.5036–0.5042 | ###
  0.5042–0.5048 | 
  0.5048–0.5054 | 
  0.5054–0.5060 | ###
  0.5060–0.5067 | ###
```

### logistic

| Statistic | Real | Null mean | p-value | Verdict |
|---|---:|---:|---:|---|
| Mean AUC | 0.4998 | 0.5001 | 0.5545 | not significant (indistinguishable from chance) |
| Precision@N | 0.5354 | 0.4965 | 0.0990 | marginal (weak evidence; not significant at 5%) |

- **Base rate** (label = 1 frequency): 0.4994
- **Bootstrap 95% CI on per-fold mean AUC**: [0.4850, 0.5141] (median 0.4996, 132 folds)

Null distribution of mean AUC over 100 label shuffles:

```
  0.4945–0.4950 | ######
  0.4950–0.4955 | 
  0.4955–0.4961 | ######
  0.4961–0.4966 | ############
  0.4966–0.4971 | ###############
  0.4971–0.4977 | ######
  0.4977–0.4982 | ##################
  0.4982–0.4988 | ###############
  0.4988–0.4993 | ##################################
  0.4993–0.4998 | #########################  <- REAL
  0.4998–0.5004 | ############################
  0.5004–0.5009 | #########################
  0.5009–0.5014 | ######################
  0.5014–0.5020 | ########################################
  0.5020–0.5025 | ##################
  0.5025–0.5030 | ######
  0.5030–0.5036 | ######
  0.5036–0.5041 | #########
  0.5041–0.5046 | 
  0.5046–0.5052 | ###############
```


## Interpretation

A low p-value means random outcomes rarely reproduce the observed ranking quality,
so the edge is unlikely to be luck. A precision@N that is significant even when the
whole-cross-section AUC is only marginal tells you the skill concentrates in the
**top of the ranking** — exactly where the strategy acts. This is an out-of-sample
research evaluation and does not guarantee future performance.
