# V8 Random Forest Experiment

## Date: December 23, 2025

## Hypothesis
Replicate academic papers claiming 80% accuracy using Random Forest with advanced features.

## Approach
- Random Forest (n=500, max_depth=20)
- XGBoost with tuned hyperparameters
- Gradient Boosting for diversity
- Ensemble of all three

## Features
- Win percentage differential
- Points differential
- Net rating differential
- Advanced stats (eFG%, TS%)
- Momentum features

## Results
| Model | Accuracy | AUC |
|-------|----------|-----|
| Random Forest | 62.8% | 0.6695 |
| XGBoost | 63.4% | 0.6674 |
| Gradient Boost | 63.1% | 0.6711 |
| Ensemble | 63.8% | 0.6747 |

**V6 Baseline: 65.4%**
**V8 Best: 63.8% (-1.6pp)**

## Key Finding
**Academic 80%+ claims are unrealistic.** They likely involve:
- Overfitting to test data
- Data leakage (using future information)
- Testing on training data
- Cherry-picked results

## Conclusion
❌ Random Forest approach is worse than XGBoost+LightGBM ensemble.

## Files
- Script: `train_v8_rf.py`
- Model: `../models/v8_nba_best.pkl`
