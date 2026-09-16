# Statistical Significance of the Selection Ranker

Generated at: 2026-07-27 13:17:23

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

- Universe: 27 tickers
- Top-N: 3 · Feature normalisation: cross-sectional rank (leakage-safe)
- Label: forward 1-month return vs cross-sectional median
- Permutations: 100

## Results

### gbm

| Statistic | Real | Null mean | p-value | Verdict |
|---|---:|---:|---:|---|
| Mean AUC | 0.5254 | 0.4984 | 0.0099 | significant at 1% (strong evidence of real skill) |
| Precision@N | 0.5445 | 0.4802 | 0.0099 | significant at 1% (strong evidence of real skill) |

- **Base rate** (label = 1 frequency): 0.4815
- **Bootstrap 95% CI on per-fold mean AUC**: [0.5043, 0.5463] (median 0.5250, 131 folds)

Null distribution of mean AUC over 100 label shuffles:

```
  0.4753–0.4775 | ####
  0.4775–0.4798 | ####
  0.4798–0.4821 | ###############
  0.4821–0.4844 | ###############
  0.4844–0.4866 | #######
  0.4866–0.4889 | ######################
  0.4889–0.4912 | #########################
  0.4912–0.4935 | ##################
  0.4935–0.4958 | #################################
  0.4958–0.4980 | ###############
  0.4980–0.5003 | ########################################
  0.5003–0.5026 | ########################################
  0.5026–0.5049 | #################################
  0.5049–0.5071 | #############################
  0.5071–0.5094 | #############################
  0.5094–0.5117 | ####
  0.5117–0.5140 | ###########
  0.5140–0.5162 | ###############
  0.5162–0.5185 | ####
  0.5185–0.5208 | ####
```

### logistic

| Statistic | Real | Null mean | p-value | Verdict |
|---|---:|---:|---:|---|
| Mean AUC | 0.5146 | 0.5006 | 0.0792 | marginal (weak evidence; not significant at 5%) |
| Precision@N | 0.5547 | 0.4797 | 0.0099 | significant at 1% (strong evidence of real skill) |

- **Base rate** (label = 1 frequency): 0.4815
- **Bootstrap 95% CI on per-fold mean AUC**: [0.4880, 0.5419] (median 0.5147, 131 folds)

Null distribution of mean AUC over 100 label shuffles:

```
  0.4798–0.4820 | #######
  0.4820–0.4842 | ###########
  0.4842–0.4864 | #######
  0.4864–0.4887 | ##################
  0.4887–0.4909 | #############################
  0.4909–0.4931 | ##################
  0.4931–0.4953 | ##################
  0.4953–0.4975 | #########################
  0.4975–0.4997 | ####################################
  0.4997–0.5020 | #########################
  0.5020–0.5042 | ##################
  0.5042–0.5064 | ########################################
  0.5064–0.5086 | ######################
  0.5086–0.5108 | ####################################
  0.5108–0.5130 | ###########
  0.5130–0.5153 | ######################  <- REAL
  0.5153–0.5175 | ###########
  0.5175–0.5197 | ####
  0.5197–0.5219 | 
  0.5219–0.5241 | ####
```


## Interpretation

A low p-value means random outcomes rarely reproduce the observed ranking quality,
so the edge is unlikely to be luck. A precision@N that is significant even when the
whole-cross-section AUC is only marginal tells you the skill concentrates in the
**top of the ranking** — exactly where the strategy acts. This is an out-of-sample
research evaluation and does not guarantee future performance.
