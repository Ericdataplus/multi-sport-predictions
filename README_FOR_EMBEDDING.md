# Project: multi-sport-predictions
Category: data-science
Path: E:\Projects\active\data-science\multi-sport-predictions

## High Level Summary (README)
# Multi-Sport Predictions

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

## Tech Context
Language: Python

## Code Structure & Key Entities
### File: `experiments\train_v10_hybrid.py`
- Function `load_data`
- Function `calculate_team_stats`
- Function `create_features`
- Function `train_stacked_model`
- Function `main`

### File: `experiments\train_v11_clash.py`
- Function `load_data`
- Function `get_behavioral_stats`
- Function `create_clash_features`
- Function `train_v11`
- Function `main`

### File: `experiments\train_v12_pure.py`
- Function `load_data`
- Function `get_pure_behavioral`
- Function `create_features`
- Function `train_model`
- Function `main`

### File: `experiments\train_v7_neural.py`
- Class `NBASeasonStatsEngine`
  - Methods: load_data, calculate_season_stats, create_features
- Class `V7NeuralNetModel`
  - Methods: build_neural_network, train_bet_style, train_all, save
- Function `train_v7_nba`

### File: `experiments\train_v8_rf.py`
- Function `load_nba_data`
- Function `calculate_advanced_features`
- Function `create_features`
- Function `train_models`
- Function `main`

### File: `experiments\train_v9_lstm.py`
- Class `NBASequenceDataset`
- Class `LSTMPredictor`
  - Methods: forward
- Function `load_and_prepare_data`
- Function `calculate_rolling_stats`
- Function `create_features`
- Function `train_model`
- Function `main`

### File: `scripts\copy_datasets.py`
- Function `find_dataset_path`
- Function `list_files`
- Function `main`

### File: `scripts\daily_predictions.py`
- Function `load_history`
- Function `save_history`
- Function `fetch_todays_games`
- Function `generate_prediction`
- Function `add_predictions_for_today`
- Function `check_and_resolve_results`
- Function `print_accuracy_report`
- Function `main`

### File: `scripts\download_datasets.py`
- Function `download_dataset`
- Function `main`

### File: `scripts\fatigue_analysis.py`
- Class `FatigueFeatureEngineer`
  - Methods: calculate_rest_days, calculate_back_to_back, calculate_games_in_window, calculate_season_fatigue, calculate_away_streak
- Function `add_fatigue_features`
- Function `train_with_fatigue_features`
- Function `analyze_all_sports`

### File: `scripts\fetch_mlb_data.py`
- Function `fetch_season_games`
- Function `fetch_multiple_seasons`
- Function `main`

### File: `scripts\fetch_nfl_data.py`
- Function `fetch_nflverse_schedules`
- Function `fetch_nflverse_team_stats`
- Function `aggregate_team_stats`
- Function `fetch_and_process_nfl_data`
- Function `main`

### File: `scripts\gather_free_data.py`
- Function `download_kaggle_datasets`
- Function `fetch_espn_historical_data`
- Function `fetch_balldontlie_data`
- Function `fetch_nhl_api_data`
- Function `summarize_data`
- Function `main`

### File: `scripts\generate_real_predictions.py`
- Function `fetch_games`
- Function `load_model`
- Function `calculate_team_features`
- Function `generate_prediction_for_game`
- Function `generate_all_predictions`

### File: `scripts\prediction_api.py`
- Function `load_model`
- Function `get_model_metrics`
- Function `home`
- Function `get_models`
- Function `predict`
- Function `get_picks`
- Function `get_parlays`
- Function `get_stats`

### File: `scripts\prediction_tracker.py`
- Function `load_history`
- Function `save_history`
- Function `add_prediction`
- Function `update_result`
- Function `get_stats`
- Function `check_espn_results`
- Function `print_summary`

### File: `scripts\run_analysis.py`
- Function `generate_sample_data`
- Function `run_sport_analysis`
- Function `main`

### File: `scripts\search_datasets.py`
- Function `main`

### File: `scripts\search_github.py`
- Function `search_github`
- Function `main`

### File: `scripts\test_all_sports_specialized.py`
- Function `load_sport_data`
- Function `get_team_stats`
- Function `create_features`
- Function `train_model`
- Function `test_tennis_moneyline`
- Function `main`

### File: `scripts\train_advanced_models.py`
- Class `AdvancedFeatureEngineer`
  - Methods: create_rolling_features, create_elo_ratings, create_efficiency_metrics
- Class `ResearchBackedTrainer`
  - Methods: load_and_prepare_data, train_model, save_models, train_all
- Function `calculate_kelly_criterion`
- Function `calculate_parlay_ev`
- Function `train_all_sports`

### File: `scripts\train_advanced_v7.py`
- Class `ELOSystem`
  - Methods: get_rating, expected_score, update, calculate_all
- Class `AdvancedV7Trainer`
  - Methods: calculate_rolling_features, train_nba_advanced, train_nfl_advanced, train_nhl_advanced, train_mlb_advanced
- Function `main`

### File: `scripts\train_all.py`
- Function `main`

### File: `scripts\train_models.py`
- Class `SportModelTrainer`
  - Methods: load_data, prepare_features, train_moneyline_model, train_spread_model, train_overunder_model
- Function `train_all_sports`

### File: `scripts\train_nba_fixed.py`
- Function `create_pregame_features`
- Function `train_nba_fixed`

### File: `scripts\train_specialized_bet_types.py`
- Class `SpecializedBetTypeTrainer`
  - Methods: load_nfl_data, load_nhl_data, load_mlb_data, load_soccer_data, create_features
- Function `train_all_sports`

### File: `scripts\train_v10_hybrid.py`
- Function `load_data`
- Function `calculate_team_stats`
- Function `create_features`
- Function `train_stacked_model`
- Function `main`

### File: `scripts\train_v11_clash.py`
- Function `load_data`
- Function `get_behavioral_stats`
- Function `create_clash_features`
- Function `train_v11`
- Function `main`

### File: `scripts\train_v12_pure.py`
- Function `load_data`
- Function `get_pure_behavioral`
- Function `create_features`
- Function `train_model`
- Function `main`

### File: `scripts\train_v13_specialized.py`
- Function `load_data`
- Function `get_team_stats`
- Function `create_features_for_bet_type`
- Function `train_specialized_model`
- Function `main`

### File: `scripts\train_v4_enhanced_test.py`
- Function `load_enhanced_data`
- Function `build_v4_features`
- Function `train_v4_with_enhanced_data`

### File: `scripts\train_v4_ensemble.py`
- Class `AdvancedFeatureEngine`
  - Methods: create_elo_features, create_momentum_features, create_rest_features
- Class `EnsembleTrainer`
  - Methods: train, train_contracts
- Function `train_sport_v4`
- Function `train_all_v4`

### File: `scripts\train_v6_behavioral.py`
- Class `BehavioralProxyFeatureEngine`
  - Methods: build_team_history, get_team_stats, calculate_fatigue_proxies, calculate_defensive_discipline, calculate_clutch_pressure
- Class `V6BehavioralProxyModel`
  - Methods: train, get_feature_importance
- Function `train_v6_basketball`

### File: `scripts\train_v6_complete.py`
- Class `BehavioralProxyEngine`
  - Methods: build_team_history, get_team_stats, calculate_behavioral_features, create_all_features
- Class `V6NBAModel`
  - Methods: train_bet_style, train_all, predict, calculate_parlay_confidence, save
- Function `load_training_data`
- Function `train_v6_complete`

### File: `scripts\train_v6_enhanced.py`
- Function `load_and_combine_data`
- Class `BehavioralProxyV6`
  - Methods: build_team_history, get_team_stats, calculate_fatigue_proxies, calculate_defensive_discipline, calculate_clutch_pressure
- Function `train_v6_enhanced`

### File: `scripts\train_v6_mlb.py`
- Class `MLBBehavioralEngine`
  - Methods: load_data, build_histories, get_team_stats, calculate_features, create_all_features
- Class `V6MLBModel`
  - Methods: train_bet_style, train_all, save
- Function `train_v6_mlb`

### File: `scripts\train_v6_nfl.py`
- Class `NFLBehavioralEngine`
  - Methods: load_data, build_histories, get_team_stats, calculate_features, create_all_features
- Class `V6NFLModel`
  - Methods: train_bet_style, train_all, save
- Function `train_v6_nfl`

### File: `scripts\train_v6_nhl.py`
- Class `NHLBehavioralEngine`
  - Methods: load_data, build_histories, get_team_stats, calculate_behavioral_features, create_all_features
- Class `V6NHLModel`
  - Methods: train_bet_style, train_all, save
- Function `train_v6_nhl`

### File: `scripts\train_v6_ou.py`
- Class `OverUnderFeatureEngine`
  - Methods: build_team_history, get_ou_stats, calculate_ou_features, create_features
- Class `V6OverUnderModel`
  - Methods: train, get_feature_importance, save
- Function `load_training_data`
- Function `train_v6_ou`

### File: `scripts\train_v6_ou_hybrid.py`
- Function `load_training_data`
- Function `build_histories`
- Function `get_stats`
- Function `create_hybrid_features`
- Function `train_hybrid`

> ... (Scanning limited to top 40 files)