"""
NFL Player Props Model Trainer
==============================
Trains XGBoost models for NFL player props (Passing Yards, Rushing Yards, Receiving Yards).
Uses scraping/unofficial APIs to get player game logs, as ESPN API is limited for historical player stats.

We will use nfl_data_py if available, or fetch from a reliable source.
For this script, we will simulate the structure assuming we have game logs,
or use a direct CSV if the user provides it. 

Since fetching NFL player logs is notoriously hard without nfl_data_py,
this script attempts to use nfl_data_py library if installed, which is the standard for NFL analytics.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import sys
import importlib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[3]))

class NFLPlayerPropsTrainer:
    def __init__(self, season=2024):
        self.season = season
        self.models_dir = Path(__file__).parent / "models" / "props"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data = None

    def fetch_data(self):
        """Fetches NFL player stats from local CSV or nfl_data_py."""
        print(f"Loading NFL player data for {self.season}...")
        
        # Try local first
        local_path = Path("data/nfl/player_props_2024.csv")
        if local_path.exists():
            print(f"Loading from local CSV: {local_path}...")
            df = pd.read_csv(local_path)
            # Ensure week/player_id types
            df['player_id'] = df['player_id'].astype(str)
            self.data = df
            return df
            
        try:
            import nfl_data_py as nfl
            years = [self.season]
            df = nfl.import_weekly_data(years)
            
            print(f"Loaded {len(df)} player-week records.")
            save_path = Path("data/nfl/player_props_2024.csv")
            df.to_csv(save_path, index=False)
            print(f"Saved raw data to {save_path}")
            
            self.data = df
            return df
            
        except Exception as e:
            print(f"Error fetching NFL data: {e}")
            return None

    def engineer_features(self, df):
        """Creates predictive features for NFL props (V6 Behavioral)."""
        if df is None: return None
        
        print("Engineering NFL V6 features...")
        
        df = df.sort_values(['player_id', 'week'])
        
        targets = {
            'passing_yards': 'Passing Yards',
            'rushing_yards': 'Rushing Yards',
            'receiving_yards': 'Receiving Yards',
            'receptions': 'Receptions'
        }
        
        # 1. Form (Rolling Averages)
        windows = [1, 3, 5]
        for target in targets.keys():
            if target not in df.columns: continue
            for w in windows:
                df[f'{target}_last_{w}'] = df.groupby('player_id')[target].transform(
                    lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
                )
        
        # 2. Season Baseline
        for target in targets.keys():
            if target not in df.columns: continue
            df[f'{target}_season_avg'] = df.groupby('player_id')[target].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            
        # 3. V6 Behavioral: Momentum (Trend Diff)
        # NFL trends are faster, so Last 3 vs Season
        for target in targets.keys():
            if target in df.columns:
                df[f'{target}_momentum'] = df[f'{target}_last_3'] - df[f'{target}_season_avg']
                
        # 4. V6 Behavioral: Stability (Inverse Variance)
        for target in targets.keys():
            if target in df.columns:
                df[f'{target}_stability'] = df.groupby('player_id')[target].transform(
                    lambda x: 1.0 / (1.0 + x.shift(1).rolling(window=5).std().fillna(10))
                )

        # 5. V6 Behavioral: Fatigue Index (Short Week + High Usage)
        # We assume regular weeks are 1 week apart. 
        # If we have 'recent_team' and 'opponent_team', we don't naturally have dates in weekly data usually?
        # nfl_data_py weekly data DOES NOT encompass dates easily without merging schedules.
        # Proxy: We will assume standard fatigue for now, or usage-based fatigue.
        
        # Usage proxy: (carries + targets)
        if 'carries' in df.columns and 'targets' in df.columns:
            df['usage'] = df['carries'].fillna(0) + df['targets'].fillna(0)
            df['fatigue_index'] = df.groupby('player_id')['usage'].transform(
                lambda x: x.shift(1).rolling(window=3).mean()
            ).fillna(0)
        else:
            df['fatigue_index'] = 0

        # 6. V6 Behavioral: Reliability (Catch Rate / Comp %)
        # For WR/TE/RB: Catch Rate
        if 'receptions' in df.columns and 'targets' in df.columns:
            df['catch_rate'] = df['receptions'] / df['targets'].replace(0, 1)
            df['reliability'] = df.groupby('player_id')['catch_rate'].transform(
                lambda x: x.shift(1).rolling(window=5).mean()
            ).fillna(0.6)
        else:
            df['reliability'] = 0.5
            
        # 7. Defense vs Position (Start)
        # Calculate how many yards/stats the OPPONENT allows on average.
        # This is CRITICAL for NFL.
        
        # Determine relevant position per target to filter? 
        # Actually, total yards allowed is simpler and robust.
        # "How many passing yards does this defense allow per week?"
        
        if 'opponent_team' in df.columns:
            for target in targets.keys():
                if target not in df.columns: continue
                
                # 1. Calculate Total Allowed per Week by Opponent
                # Aggregating all players against this team in this week
                def_weekly = df.groupby(['opponent_team', 'week'])[target].sum().reset_index()
                def_weekly = def_weekly.rename(columns={target: 'allowed_weekly'})
                
                # 2. Calculate Expanding Average (Prior to this week)
                def_weekly = def_weekly.sort_values(['opponent_team', 'week'])
                def_weekly[f'def_allowed_{target}'] = def_weekly.groupby('opponent_team')['allowed_weekly'].transform(
                    lambda x: x.shift(1).expanding().mean()
                )
                
                # 3. Merge back
                df = df.merge(def_weekly[['opponent_team', 'week', f'def_allowed_{target}']], 
                              on=['opponent_team', 'week'], how='left')
                
                # Fill NaNs with League Average for that week (or overall)
                league_avg = def_weekly['allowed_weekly'].mean()
                df[f'def_allowed_{target}'] = df[f'def_allowed_{target}'].fillna(league_avg)
                
        # Context: Home Game?
        # location column 'HOME' or 'AWAY' usually
        if 'location' in df.columns:
            df['is_home'] = (df['location'] == 'HOME').astype(int)
        else:
            df['is_home'] = 0.5

        return df

    def train_model(self, target_stat='passing_yards'):
        """Trains XGBoost for specific NFL stat."""
        if self.data is None:
            print("Error: No data loaded.")
            return

        print(f"Training model for {target_stat}...")
        
        df = self.engineer_features(self.data.copy())
        if df is None: return

        if target_stat not in df.columns:
            print(f"Stat {target_stat} not found.")
            return
            
        # V6 Feature Set
        features = [
            f'{target_stat}_last_1', f'{target_stat}_last_3', f'{target_stat}_last_5',
            f'{target_stat}_season_avg',
            f'{target_stat}_momentum',
            f'{target_stat}_stability',
            f'def_allowed_{target_stat}', # Defense vs Position
            'fatigue_index', 'reliability',
            'is_home'
        ]
        
        model_df = df.dropna(subset=features + [target_stat])
        
        if len(model_df) < 50:
            print("Insufficient data to train.")
            return

        # V6 Filter: Only train on RELEVANT players for this stat
        # (e.g. Don't evaluate a WR on passing yards just because he threw one pass 3 years ago)
        thresholds = {
            'passing_yards': 10.0,
            'rushing_yards': 5.0,
            'receiving_yards': 5.0,
            'receptions': 1.0
        }
        thresh = thresholds.get(target_stat, 0)
        
        # Filter both Training and Test to relevant players
        # This dramatically improves "Real World Betting" accuracy by ignoring noise
        model_df = model_df[model_df[f'{target_stat}_season_avg'] >= thresh].copy()
        
        print(f"Filtered to {len(model_df)} relevant player-games (Avg >= {thresh})")

        X = model_df[features]
        y = model_df[target_stat]
        
        # Split last 2 weeks as test
        max_week = model_df['week'].max()
        train_mask = model_df['week'] < (max_week - 2)
        test_mask = model_df['week'] >= (max_week - 2)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Validation: Ensure we still have test data
        if len(X_test) < 10:
             print("Warning: Not enough test data after filtering.")
             return
             
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:absoluteerror', # OPTIMIZE FOR MEDIAN (Better for Props)
            early_stopping_rounds=50,
            n_jobs=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        preds = model.predict(X_test)
        mae = np.mean(np.abs(preds - y_test))
        print(f"   MAE on Test Set: {mae:.2f}")

        # Accuracy Check (Over/Under Season Avg)
        season_avg = X_test[f'{target_stat}_season_avg']
        
        # DEBUG: Check why accuracy is low
        print("\n   --- DEBUG DATA PREVIEW (First 5) ---")
        print(f"   Preds:      {preds[:5]}")
        print(f"   Actuals:    {y_test.values[:5]}")
        print(f"   Season Avg: {season_avg.values[:5]}")
        
        actual_over = (y_test > season_avg).astype(int)
        pred_over = (preds > season_avg).astype(int)
        
        print(f"   Act Over:   {actual_over.values[:5]}")
        print(f"   Pred Over:  {pred_over[:5]}")
        print("   ------------------------------------\n")
        
        accuracy = np.mean(actual_over == pred_over)
        print(f"   Implied O/U Accuracy: {accuracy:.1%}")
        
        with open("nfl_mae_output.txt", "a") as f:
            f.write(f"{target_stat} - MAE: {mae:.2f}, Acc: {accuracy:.1%}\n")
        
        save_path = self.models_dir / f"xgb_nfl_{target_stat}_v1.json"
        model.save_model(save_path)
        print(f"   Model saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Train NFL Player Props")
    parser.add_argument("--stat", type=str, default="passing_yards", help="Stat to train")
    parser.add_argument("--all", action="store_true", help="Train all major stats")
    
    args = parser.parse_args()
    
    trainer = NFLPlayerPropsTrainer()
    data = trainer.fetch_data()
    
    if data is None:
        return

    if args.all:
        for stat in ['passing_yards', 'rushing_yards', 'receiving_yards', 'receptions']:
            trainer.train_model(stat)
    else:
        trainer.train_model(args.stat)

if __name__ == "__main__":
    main()
