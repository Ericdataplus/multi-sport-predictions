# V10 Multi-Window Stacking Experiment

## Date: December 23, 2025

## Hypothesis
Use multiple rolling windows (5, 10, 20, 40 games) to capture both short-term form and long-term trends, then stack multiple models with a meta-learner.

## Approach
- Extended rolling windows: 5, 10, 20, 40 games
- XGBoost + LightGBM + GradientBoosting (Level 1)
- Logistic Regression meta-learner (Level 2)
- 5-fold TimeSeriesSplit for out-of-fold predictions

## Features (31 total)
- Win/Net/Pts differentials for each window (4 windows × 4 features)
- Trend features (momentum short/long)
- Raw stats for context
- Simulated market features

## Results
| Bet Type | V6 | V10 | Change |
|----------|-----|-----|--------|
| Moneyline | 65.4% | 61.0% | -4.4pp |
| Spread | 73.4% | 56.7% | -16.7pp |
| Total | 55.0% | 62.6% | +7.6pp |

Individual model contributions:
- XGB: 62.6%
- LGB: 62.1%  
- GB: 60.4%

## Key Finding
**More windows ≠ better prediction.** The additional complexity added noise rather than signal. Stacking with a meta-learner didn't improve results.

## Lessons Learned
1. Simple 10-20 game windows work best
2. Stacking is overhead without benefit for this problem
3. XGBoost alone is nearly as good as ensembles
4. More features can hurt generalization

## Conclusion
❌ Multi-window stacking is worse than V6. Stick to simple approach.

## Files
- Script: `train_v10_hybrid.py`
- Model: `../models/v10_nba_stacked.pkl`
