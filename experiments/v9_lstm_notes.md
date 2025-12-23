# V9 LSTM + Embeddings Experiment

## Date: December 23, 2025

## Hypothesis
Replicate the 72.35% accuracy LSTM paper from arXiv by using:
- Long-sequence LSTM (inspired by 9,840 game sequences)
- Team embeddings (learned representations for each team)
- Multi-task learning (predict moneyline, spread, total simultaneously)
- Attention mechanism

## Architecture
```python
LSTMPredictor(
    home_embedding = Embedding(30 teams, 32 dim)
    away_embedding = Embedding(30 teams, 32 dim)
    lstm = BiLSTM(hidden=128, layers=2, bidirectional=True)
    attention = Linear → Tanh → Linear
    fc = Dense(128→64) with Dropout(0.3)
    heads = [moneyline, spread, total]
)
```

## Training
- Epochs: 100 (early stopped at 16)
- Optimizer: AdamW (lr=0.001, weight_decay=0.01)
- Loss: BCE (weighted toward moneyline)
- GPU: CUDA

## Results
| Bet Type | V6 | V9 | Change |
|----------|-----|-----|--------|
| Moneyline | 65.4% | 55.2% | -10.2pp |
| Spread | 73.4% | 55.1% | -18.3pp |
| Total | 55.0% | 64.0% | +9.0pp |

## Key Finding
**LSTM severely underperformed.** Reasons:
1. Tabular data ≠ sequential data (LSTM overhead without benefit)
2. Model underfitting (early stopped at epoch 16)
3. Paper's 72.35% likely uses different methodology or has leakage
4. Team embeddings didn't capture useful patterns

## Lessons Learned
- Deep learning is NOT the answer for tabular sports prediction
- XGBoost dominates on structured tabular data
- Don't trust academic claims without replication

## Conclusion
❌ LSTM approach is MUCH worse than XGBoost. Do not pursue deep learning for this problem.

## Files
- Script: `train_v9_lstm.py`
- Model: `../models/v9_nba_lstm.pkl`
