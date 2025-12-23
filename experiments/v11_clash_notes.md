# V11 Behavioral Clash Experiment

## Date: December 23, 2025

## Hypothesis
Games are won in MISMATCHES, not absolute strength. Model how behaviors INTERACT rather than just comparing stats.

## Approach
Create "clash" features that capture matchup dynamics:
- Steal Rate × Opponent Turnover Rate = exploitation potential
- Block Rate × (1 - Opponent FG%) = rim protection matchup
- Pace differential = tempo control
- Assist Rate / Opponent Steal Rate = playmaking edge
- Foul rate differential = discipline clash

## Features (25 total)
- 6 Clash interaction features
- 3 Trend features (momentum)
- 4 Context features (win%, net rating, sample size)
- 12 Raw behavioral stats

## Results
| Model | Accuracy | AUC |
|-------|----------|-----|
| XGBoost | 59.7% | - |
| LightGBM | 60.4% | - |
| Ensemble | 59.8% | 0.6301 |

**V6: 65.4% → V11: 59.8% (-5.6pp)**

## Feature Importance (Top 5)
1. net_rating_diff (19.5%) ← Outcome-based!
2. win_pct_diff (16.8%) ← Outcome-based!
3. steal_exploit_diff (4.7%) ← Clash feature
4. playmaking_edge (3.1%) ← Clash feature
5. steal_exploit_home (3.0%) ← Clash feature

## Key Finding
**The model still relies on outcome features (net rating, win%).** The behavioral clash features are used but not dominant. XGBoost is already smart enough to learn interactions - we don't need to pre-compute them.

## Lessons Learned
1. Don't pre-compute feature interactions - let XGBoost learn them
2. Outcome features (win%, net rating) are highly predictive
3. Complex feature engineering adds complexity without benefit

## Conclusion
❌ Behavioral clash approach is worse than V6. The insight (games are won in mismatches) is correct, but XGBoost already captures this from raw features.

## Files
- Script: `train_v11_clash.py`
- Model: `../models/v11_nba_clash.pkl`
