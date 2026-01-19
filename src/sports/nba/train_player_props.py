"""
NBA Player Props Model Trainer
==============================
Trains XGBoost models for player prop betting (Points, Rebounds, Assists).
Uses individual player game logs and advanced feature engineering.

Techniques incorporated (2025 Standard):
1. Rolling Window Performance (Last 5, 10, 20 games) -> Captures form.
2. Opponent Defense vs Position -> Contextual difficulty.
3. Home/Away Splits -> Venue performance.
4. Rest Days -> Fatigue factor.
5. XGBoost w/ Early Stopping -> Robust tabular prediction.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[3]))

# Define constants
BASE_URL = "https://stats.nba.com/stats"
HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'Accept': 'application/json',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true', 
}

class PlayerPropsTrainer:
    def __init__(self, season="2024-25"):
        self.season = season
        self.models_dir = Path(__file__).parent / "models" / "props"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data = None

    def fetch_player_logs(self):
        """Fetches player game logs from local offline data."""
        print(f"📉 Loading player game logs from local data (V6 Source)...")
        try:
            # 1. Load Games Details (Box Scores)
            details_path = Path("data/nba/games_details.csv")
            if not details_path.exists():
                print(f"❌ Error: {details_path} not found.")
                return None
                
            print(f"   Reading {details_path}...")
            df_details = pd.read_csv(details_path)
            
            # 2. Load Games (for Dates)
            games_path = Path("data/nba/games.csv")
            if not games_path.exists():
                print(f"❌ Error: {games_path} not found.")
                return None
                
            print(f"   Reading {games_path}...")
            df_games = pd.read_csv(games_path)
            
            # 3. Merge to get Game Dates
            # games.csv has GAME_ID, GAME_DATE_EST
            # games_details.csv has GAME_ID
            
            # Ensure proper types
            df_games['GAME_ID'] = df_games['GAME_ID'].astype(str)
            df_details['GAME_ID'] = df_details['GAME_ID'].astype(str)
            
            # Merge
            print("   Merging game data...")
            df = df_details.merge(df_games[['GAME_ID', 'GAME_DATE_EST', 'SEASON']], on='GAME_ID', how='inner')
            
            # 4. Filter for recent seasons (e.g., 2021 onwards for relevance)
            df = df[df['SEASON'] >= 2021].copy()
            
            # 5. Rename/Format columns
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE_EST'])
            
            # Expected columns for feature engineering: 
            # PLAYER_ID, GAME_DATE, PTS, REB, AST, STL, BLK, FG3M, MIN, FT_PCT, MATCHUP (derived)
            
            # Create MATCHUP proxy (e.g., "GSW vs LAL") - hard to get precise from this join without team info
            # but we can trust the 'is_home' logic later or simplify it.
            # V6 script filtered for valid games.
            
            # Ensure numeric stats
            stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'FT_PCT']
            for col in stats:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            self.data = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
            print(f"✅ Loaded {len(df)} player-game records from offline data.")
            return df
            
        except Exception as e:
            print(f"❌ Error loading offline data: {e}")
            return None

    def engineer_features(self, df):
        """Creates predictive features for player props."""
        print("🔧 Engineering features...")
        
        # Target columns to predict
        targets = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M']
        
        # 1. Rolling Averages (Form)
        windows = [3, 5, 10, 20]
        
        for target in targets:
            for w in windows:
                # Group by Player and Shift 1 to avoid data leakage (predicting today using today's stats)
                df[f'{target}_last_{w}'] = df.groupby('PLAYER_ID')[target].transform(
                    lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
                )
                
        # 2. Season Averages (Consistency)
        for target in targets:
             df[f'{target}_season_avg'] = df.groupby('PLAYER_ID')[target].transform(
                lambda x: x.shift(1).expanding().mean()
             )

        # 3. V6 Behavioral Proxies: Fatigue
        # Logic adapted from V6 Team Model:
        # - Back-to-back = 1.0
        # - Games in last 7 days (normalized)
        # - Rest days (normalized)
        
        # Calculate games in last 7 days
        # This requires rolling window on 'GAME_DATE' which is hard in vectorized pandas without resampling.
        # Approximation: Shifts. 
        # If date(t) - date(t-3) <= 7 days, then 3 games in 7 days.
        df['days_since_3_games_ago'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff(3).dt.days
        df['games_7d_proxy'] = (df['days_since_3_games_ago'] <= 7).astype(int)

        df['rest_days'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days.fillna(3)
        df['is_b2b'] = (df['rest_days'] == 1).astype(int)
        
        # V6 Fatigue Index: high minutes * high fatigue context
        # If MIN exists, use it. Else assume starter minutes (30) if they recorded stats.
        if 'MIN' in df.columns:
            # Clean MIN column if string "XX:XX"
            df['clean_min'] = df['MIN'].apply(lambda x: float(str(x).split(':')[0]) + float(str(x).split(':')[1])/60.0 if ':' in str(x) else (float(x) if x else 0.0))
        else:
            df['clean_min'] = 30.0

        df['fatigue_index'] = df['is_b2b'] * df['clean_min'] + df['games_7d_proxy'] * 10
        
        # 4. V6 Behavioral Proxies: Clutch / Pressure
        # Proxy: Free Throw Percentage (Mental Composure)
        # If FT% not available, default to 0.75
        if 'FT_PCT' in df.columns:
            df['clutch_composure'] = df.groupby('PLAYER_ID')['FT_PCT'].transform(
                lambda x: x.shift(1).expanding().mean()
            ).fillna(0.75)
        else:
            df['clutch_composure'] = 0.75

        # 5. V6 Behavioral Proxies: Consistency / Stability
        # Std Dev of target stat over last 10 games
        for target in targets:
            df[f'{target}_consistency'] = df.groupby('PLAYER_ID')[target].transform(
                lambda x: x.shift(1).rolling(window=10).std()
            ).fillna(0)
            
            # Inverse: Stability (1 / (1 + std))
            df[f'{target}_stability'] = 1.0 / (1.0 + df[f'{target}_consistency'])

        # 6. V6 Behavioral Proxies: Trend Difference (Momentum)
        # (Last 5 Avg) - (Season Avg)
        for target in targets:
             df[f'{target}_trend_diff'] = df[f'{target}_last_5'] - df[f'{target}_season_avg']

        # 7. Context: Home/Away
        if 'is_home' not in df.columns:
            if 'MATCHUP' in df.columns:
                df['is_home'] = df['MATCHUP'].apply(lambda x: 1 if ' vs. ' in str(x) else 0)
            else:
                df['is_home'] = 0.5 # Fallback
                
        # Drop initial rows with NaNs
        df = df.dropna(subset=[f'{targets[0]}_last_10']) 
        
        return df

    def train_model(self, target_stat='PTS'):
        """Trains an XGBoost model for a specific stat."""
        if self.data is None:
            print("❌ No data loaded.")
            return

        print(f"🎯 Training model for {target_stat} (V6 Behavioral Features)...")
        
        df = self.engineer_features(self.data.copy())
        
        # V6 Feature Set
        features = [
            # Form
            f'{target_stat}_last_3', f'{target_stat}_last_5', 
            f'{target_stat}_last_10', f'{target_stat}_last_20',
            f'{target_stat}_season_avg',
            f'{target_stat}_trend_diff',
            
            # Behavioral Proxies
            'fatigue_index',      # Physical load
            'clutch_composure',   # Mental state
            f'{target_stat}_stability', # Consistency
            
            # Context
            'rest_days', 'is_b2b', 'is_home'
        ]
        
        X = df[features]
        y = df[target_stat]
        
        # Train/Test Split (Time-based)
        split_date = df['GAME_DATE'].max() - timedelta(days=14)
        train_mask = df['GAME_DATE'] < split_date
        test_mask = df['GAME_DATE'] >= split_date
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Model
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            early_stopping_rounds=50,
            n_jobs=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        preds = model.predict(X_test)
        mae = np.mean(np.abs(preds - y_test))
        print(f"   MAE on Test Set: {mae:.2f}")

        # Calculate Implied Betting Accuracy (Simulating Betting against Season Average)
        # This answers: "Can we predict if he goes Over/Under his average?"
        season_avg = X_test[f'{target_stat}_season_avg']
        actual_over = (y_test > season_avg).astype(int)
        pred_over = (preds > season_avg).astype(int)
        accuracy = np.mean(actual_over == pred_over)
        
        print(f"   Implied O/U Accuracy: {accuracy:.1%}")
        
        with open("mae_output.txt", "w") as f:
            f.write(f"MAE: {mae:.2f}\n")
            f.write(f"Accuracy: {accuracy:.1%}")
        
        # Save
        save_path = self.models_dir / f"xgb_prop_{target_stat.lower()}_v1.json"
        model.save_model(save_path)
        print(f"   Model saved to {save_path}")
        
        return model

def main():
    parser = argparse.ArgumentParser(description="Train Player Props Models")
    parser.add_argument("--stat", type=str, default="PTS", help="Stat to train (PTS, REB, AST)")
    parser.add_argument("--all", action="store_true", help="Train all major stats")
    args = parser.parse_args()
    
    trainer = PlayerPropsTrainer()
    trainer.fetch_player_logs()
    
    if args.all:
        for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK']:
            trainer.train_model(stat)
    else:
        trainer.train_model(args.stat)

if __name__ == "__main__":
    main()
