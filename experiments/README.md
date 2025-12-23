# Experimental Models Archive

This folder contains past model experiments and their results. Use this to avoid repeating failed approaches and build on what worked.

## 📊 Summary of All Experiments (December 2025)

| Model | Approach | NBA Moneyline | Result | Verdict |
|-------|----------|---------------|--------|---------|
| **V6 (BEST)** | XGBoost+LGB + Behavioral Proxy | **65.4%** | Baseline | ✅ Use this |
| V7 | Neural Network (kyleskom-style) | 65.2% | -0.2pp | ❌ No improvement |
| V8 | Random Forest | 63.8% | -1.6pp | ❌ Worse |
| V9 | LSTM + Team Embeddings | 55.2% | -10.2pp | ❌ Much worse |
| V10 | Multi-Window Stacking | 61.0% | -4.4pp | ❌ Worse |
| V11 | Behavioral Clash Features | 59.8% | -5.6pp | ❌ Worse |
| V12 | Pure Behavioral (no outcomes) | 58.0% | -7.4pp | ❌ Worse |

## 🏆 What Works (V6 Approach)

### Model Architecture
- XGBoost + LightGBM ensemble (50/50 averaging)
- Isotonic calibration for probability estimates
- Time-based train/test split (no future leakage)

### Features That Matter
1. **Outcome features** (win%, net rating) - encodes team strength
2. **Behavioral proxy features** (steals, blocks, turnovers, fouls, assists)
3. **Rolling windows** (10-20 games optimal)
4. **Simple differentials** (home - away)

### Hyperparameters
```python
xgb.XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.02,
    reg_lambda=5.0, reg_alpha=1.0,
    subsample=0.8, colsample_bytree=0.7,
)
```

## ❌ What Doesn't Work

### 1. Deep Learning (V9)
- LSTMs underfit on tabular sports data
- Neural networks need more data than we have
- Team embeddings didn't capture useful patterns
- Result: 55.2% (worse than random with home advantage)

### 2. Complex Feature Engineering (V10, V11)
- Multi-window features (5/10/20/40 games) added noise
- "Clash" features (steal vs turnover interactions) didn't help
- XGBoost already learns interactions - don't pre-compute them
- Result: 59-61% (worse than simple features)

### 3. Removing Outcome Features (V12)
- Pure behavioral stats alone are not enough
- Win% and net rating encode important signal
- Result: 58% (7pp worse)

### 4. Stacking (V10)
- Meta-learner didn't improve over simple averaging
- Added complexity without benefit

### 5. Replicating External Claims (V7, V8)
- kyleskom's 69% NBA claim was likely cherry-picked
- Academic 80% claims are likely overfitting/leakage
- True ceiling appears to be ~65-66% for public data

## 🔮 Promising Directions (NOT YET TRIED)

### 1. Betting Line Features ⭐ High Priority
- Use opening spreads/totals as features
- Market encodes information we don't have
- Free data available on Kaggle
- Expected gain: +3-5pp

### 2. Injury/Rest Analysis
- Systematic back-to-back detection
- Travel distance features
- Recent minutes load
- Expected gain: +1-2pp

### 3. Real-Time Lineup Data
- Scrape injury reports before games
- Starting lineup confirmations
- Expected gain: Unknown but valuable

### 4. Different Sports
- Soccer spread model is 75.3% - very strong
- NHL needs better data
- Tennis could be improved

## 📁 Experiment Files

Each experiment has:
1. Training script in `scripts/train_vX_*.py`
2. Model artifact in `models/vX_*.pkl`
3. Metrics in `models/vX_*_metrics.json`

## 🧠 Key Learnings for Future AI

1. **Don't try deep learning again** - XGBoost dominates for tabular sports data
2. **Don't remove outcome features** - they're predictive
3. **Don't over-engineer features** - let XGBoost learn interactions
4. **Focus on data acquisition** - betting lines, injuries, lineups
5. **The 65% ceiling is real** without privileged data
6. **Validate external claims** - most are overstated

## 📅 Timeline

- **Dec 22, 2025**: V6 baseline established (65.4%)
- **Dec 23, 2025**: V7-V12 experiments completed
- **Conclusion**: V6 is near-optimal for available data

---

*Last updated: December 23, 2025*
