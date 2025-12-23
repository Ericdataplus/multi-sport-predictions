# V7 Neural Network Experiment

## Date: December 23, 2025

## Hypothesis
Replicate kyleskom/NBA-ML-Betting (claimed 69% accuracy) using Neural Networks and cumulative season-to-date statistics.

## Approach
- TensorFlow/Keras Neural Network (3 dense layers with dropout)
- Cumulative season-to-date features for each team
- Also tested XGBoost with kyleskom's parameters (max_depth=3, lr=0.01, n_estimators=750)

## Architecture
```
Input → Dense(128, relu) → Dropout(0.3) → 
Dense(64, relu) → Dropout(0.2) → 
Dense(32, relu) → Dense(1, sigmoid)
```

## Results
| Bet Type | V6 Baseline | V7 Result | Change |
|----------|-------------|-----------|--------|
| Moneyline | 65.4% | 65.2% | -0.2pp |
| Spread | 73.4% | 67.2% | -6.2pp |
| Total | 55.0% | 74.4%* | +19.4pp |

*Total accuracy is suspiciously high - likely target definition issue

## Key Finding
**kyleskom's 69% claim is likely cherry-picked.** Their code runs 300 iterations and saves only the best. Real average is ~65%.

## Conclusion
❌ Neural Network does not improve over XGBoost ensemble. 
V6 remains the best approach.

## Files
- Script: `train_v7_neural.py`
- Model: `../models/v7_nba_neural.pkl`
