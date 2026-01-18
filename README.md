# Jig Sports Predictions


ML-powered sports prediction dashboard with V6 behavioral proxy models.

## Supported Sports
- 🏀 NBA (65% ML, 73% Spread)
- 🏈 NFL (65% ML, 65% Spread)
- ⚽ Soccer (64% ML, 75% Spread)
- 🏒 NHL (51% ML, 59% Spread)
- ⚾ MLB (53% ML, 56% Spread)

## Features
- V6 XGBoost + LightGBM ensemble models
- Behavioral proxy features (fatigue, discipline, form)
- Moneyline, Spread, O/U, and Contracts predictions
- Parlay builder with confidence calculations
- Prediction history tracking

## Quick Start
```bash
# Start dashboard
npx http-server -p 8085

# Open browser
http://127.0.0.1:8085
```

## Training Models
```bash
python scripts/train_v6_nba.py
python scripts/train_v6_nfl.py
python scripts/train_v6_soccer.py
python scripts/train_v6_nhl.py
python scripts/train_v6_mlb.py
```

## Data Sources (Free, No API Keys)
- NBA: NBA.com Stats API
- NFL: Spreadspoke CSV
- Soccer: Transfermarkt CSV
- NHL: Kaggle CSV
- MLB: MLB Stats API

## License
Private - All Rights Reserved

## Credits
Original framework based on open source work by Ericdataplus.
