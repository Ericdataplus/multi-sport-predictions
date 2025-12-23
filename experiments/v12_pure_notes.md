# V12 Pure Behavioral Experiment

## Date: December 23, 2025

## Hypothesis
Outcome-based features (wins, points) might add NOISE because they're the result of luck + skill. Pure behavioral features (steals, blocks, turnovers, FG%) might generalize better because they measure underlying SKILL directly.

## Approach
Remove ALL outcome-based features:
- NO win percentage
- NO points scored
- NO net rating
- ONLY: FG%, FG3%, FT%, rebounds, assists, steals, blocks, turnovers, fouls

## Features (25 total)
- Shooting skill differentials (FG%, FG3%, FT%)
- Hustle differentials (rebounds, assists)
- Defensive intensity (steals, blocks) - simulated
- Ball security (turnovers) - simulated
- Discipline (fouls) - simulated
- Raw stats for both teams

## Results
| Model | Accuracy | AUC |
|-------|----------|-----|
| XGBoost | 58.3% | - |
| LightGBM | 57.6% | - |
| Ensemble | 58.0% | 0.5904 |

**V6: 65.4% → V12: 58.0% (-7.4pp)**

## Feature Importance (Top 5)
1. fg_pct_diff (11.1%)
2. fg3_pct_diff (5.9%)
3. home_fg_pct (5.2%)
4. reb_diff (5.0%)
5. away_fg_pct (4.5%)

## Key Finding
**Outcome features ARE predictive.** Win% and net rating encode important signal that pure behavioral stats don't capture. The "noise" theory was incorrect - outcomes contain real information about team strength.

## Lessons Learned
1. Don't remove outcome features - they're predictive
2. Behavioral features work BEST when combined with outcomes
3. V6's approach (both behavioral + outcome) is correct

## Conclusion
❌ Pure behavioral approach is significantly worse than V6. Outcome features are necessary.

## Files
- Script: `train_v12_pure.py`
- No model saved (experimental only)
