"""
Generate V6 Behavioral Player Props
===================================
Uses the offline V6 XGBoost models to generate player prop predictions 
for the most recent games in the dataset.
Saves to data/player_props_predictions.json for the dashboard.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "src/sports"
OUTPUT_FILE = DATA_DIR / "player_props_predictions.json"

def load_nba_data():
    """Load and prepare NBA data for inference."""
    print("🏀 Loading NBA Data...")
    try:
        details = pd.read_csv(DATA_DIR / "nba/games_details.csv")
        games = pd.read_csv(DATA_DIR / "nba/games.csv")
        
        # Merge to get dates
        games['GAME_ID'] = games['GAME_ID'].astype(str)
        details['GAME_ID'] = details['GAME_ID'].astype(str)
        
        df = details.merge(games[['GAME_ID', 'GAME_DATE_EST', 'SEASON']], on='GAME_ID', how='inner')
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE_EST'])
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
        
        return df
    except Exception as e:
        print(f"Error loading NBA data: {e}")
        return None

def engineer_features_nba(df):
    """Replicate V6 NBA Feature Engineering."""
    targets = ['PTS', 'REB', 'AST', 'STL', 'BLK']
    
    # Ensure numerics
    for col in targets + ['MIN', 'FG3M', 'FT_PCT']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # 1. Rolling Stats
    windows = [3, 5, 10, 20]
    for target in targets:
        for w in windows:
            df[f'{target}_last_{w}'] = df.groupby('PLAYER_ID')[target].transform(
                lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
            )
            
    # 2. Season Avg
    for target in targets:
        df[f'{target}_season_avg'] = df.groupby('PLAYER_ID')[target].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        
    # 3. Behavioral: Fatigue
    df['days_since_3_games_ago'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff(3).dt.days
    df['games_7d_proxy'] = (df['days_since_3_games_ago'] <= 7).astype(int)
    df['rest_days'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days.fillna(3)
    df['is_b2b'] = (df['rest_days'] == 1).astype(int)
    
    if 'MIN' in df.columns:
        df['clean_min'] = df['MIN'].apply(lambda x: float(str(x).split(':')[0]) + float(str(x).split(':')[1])/60.0 if ':' in str(x) else (float(x) if x else 0.0))
    else:
        df['clean_min'] = 30.0
        
    df['fatigue_index'] = df['is_b2b'] * df['clean_min'] + df['games_7d_proxy'] * 10
    
    # 4. Behavioral: Clutch
    if 'FT_PCT' in df.columns:
        df['clutch_composure'] = df.groupby('PLAYER_ID')['FT_PCT'].transform(
            lambda x: x.shift(1).expanding().mean()
        ).fillna(0.75)
    else:
        df['clutch_composure'] = 0.75
        
    # 5. Stability & Trend
    for target in targets:
        df[f'{target}_consistency'] = df.groupby('PLAYER_ID')[target].transform(
            lambda x: x.shift(1).rolling(window=10).std()
        ).fillna(0)
        df[f'{target}_stability'] = 1.0 / (1.0 + df[f'{target}_consistency'])
        
        # Momentum
        if f'{target}_last_5' in df.columns and f'{target}_season_avg' in df.columns:
            df[f'{target}_trend_diff'] = df[f'{target}_last_5'] - df[f'{target}_season_avg']
        else:
            df[f'{target}_trend_diff'] = 0

    # 6. Context: Home/Away
    # Need simplistic checking if no MATCHUP
    if 'MATCHUP' in df.columns:
        df['is_home'] = df['MATCHUP'].apply(lambda x: 1 if ' vs. ' in str(x) else 0)
    else:
        df['is_home'] = 0.5
        
    return df

def generate_nba_predictions(df):
    """Generate NBA predictions for latest games."""
    if df is None: return []
    
    print("🔧 Engineering NBA Features...")
    df = engineer_features_nba(df)
    
    # Filter to most recent date in DB to simulate "Today"
    latest_date = df['GAME_DATE'].max()
    today_df = df[df['GAME_DATE'] == latest_date].copy()
    
    print(f"🔮 Predicting for {len(today_df)} players on {latest_date.strftime('%Y-%m-%d')}...")
    
    predictions = []
    stats_map = {
        'PTS': 'Points',
        'REB': 'Rebounds',
        'AST': 'Assists'
    }
    
    for stat, label in stats_map.items():
        model_path = MODELS_DIR / f"nba/models/props/xgb_prop_{stat.lower()}_v1.json"
        if not model_path.exists():
            continue
            
        print(f"   Loading model: {model_path.name}")
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        
        # Prepare Features (Must match training exactly)
        features = [
            f'{stat}_last_3', f'{stat}_last_5', 
            f'{stat}_last_10', f'{stat}_last_20',
            f'{stat}_season_avg',
            f'{stat}_trend_diff',
            'fatigue_index',
            'clutch_composure',
            f'{stat}_stability',
            'rest_days', 'is_b2b', 'is_home'
        ]
        
        # Predict
        valid_rows = today_df.dropna(subset=features)
        if len(valid_rows) == 0: continue
            
        preds = model.predict(valid_rows[features])
        
        for idx, (i, row) in enumerate(valid_rows.iterrows()):
            projection = preds[idx]
            line = row[f'{stat}_season_avg']
            if pd.isna(line) or line == 0: continue
            
            # Smart Line: Round to nearest 0.5
            line = round(line * 2) / 2
            
            # Pick & Confidence
            gap_val = abs(projection - line)
            if projection > line:
                pick = "OVER"
            else:
                pick = "UNDER"
                
            # Dynamic Confidence Calculation (V6 Enhanced)
            # 1. Base Accuracy (Conservative start)
            base_acc = 0.58
            
            # 2. Gap Strength (Percentage Diff)
            gap_pct = gap_val / max(line, 0.1)
            
            # 3. Dampening for Low Lines (Small numbers have huge % swings)
            if line < 3.0: gap_pct *= 0.3
            elif line < 10.0: gap_pct *= 0.6
            elif line < 25.0: gap_pct *= 0.8
            
            # 4. Calculate Confidence
            # Formula: Base + (Gap * Scaling)
            # e.g. 20% gap on points -> 0.58 + (0.2 * 0.4) = 0.66
            conf = base_acc + (gap_pct * 0.4)
            
            # 5. Cap limits
            conf = min(max(conf, 0.53), 0.82)
            
            # Data Cleaning
            pos = row.get('START_POSITION')
            if not pos: pos = 'Res'
            
            predictions.append({
                "player": row.get('PLAYER_NAME', 'Unknown'),
                "team": row.get('TEAM_ABBREVIATION', 'NBA'),
                "prop": label,
                "line": float(line),
                "pick": pick,
                "confidence": float(round(conf, 2)),
                "trend": "🔥 Hot" if row.get(f'{stat}_trend_diff', 0) > 2 else "❄️ Cold" if row.get(f'{stat}_trend_diff', 0) < -2 else "➡️ Neutral",
                "player_avg": float(round(row[f'{stat}_season_avg'], 1)),
                "event_group": "Regular Season",
                "matchup": str(row.get('MATCHUP', 'NBA Game')),
                "sport": "nba",
                "position": pos
            })
            
    return predictions

def load_nfl_data():
    """Load NFL data."""
    print("🏈 Loading NFL Data...")
    path = DATA_DIR / "nfl/player_props_2024.csv"
    if not path.exists(): return None
    return pd.read_csv(path)

def generate_nfl_predictions(df):
    """Generate NFL predictions."""
    if df is None: return []
    
    print("🔧 Engineering NFL Features...")
    df = df.sort_values(['player_id', 'week'])
    
    stats_map = {
        'passing_yards': 'Passing Yds',
        'rushing_yards': 'Rushing Yds',
        'receiving_yards': 'Receiving Yds',
        'receptions': 'Receptions'
    }
    
    # --- Feature Engineering (Short Version) ---
    # 1. Form
    for stat in stats_map.keys():
        if stat not in df.columns: continue
        for w in [1, 3, 5]:
            df[f'{stat}_last_{w}'] = df.groupby('player_id')[stat].transform(
                lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
            )
        # Season Avg
        df[f'{stat}_season_avg'] = df.groupby('player_id')[stat].transform(
             lambda x: x.shift(1).expanding().mean()
        )
        # Momentum
        if f'{stat}_last_3' in df.columns and f'{stat}_season_avg' in df.columns:
            df[f'{stat}_momentum'] = df[f'{stat}_last_3'] - df[f'{stat}_season_avg']
        else:
            df[f'{stat}_momentum'] = 0
            
        # Stability
        df[f'{stat}_stability'] = df.groupby('player_id')[stat].transform(
            lambda x: 1.0 / (1.0 + x.shift(1).rolling(window=5).std().fillna(10))
        )
        
        # Defense vs Position (Simulated/Calculated)
        if 'opponent_team' in df.columns:
            def_weekly = df.groupby(['opponent_team', 'week'])[stat].sum().reset_index()
            def_weekly = def_weekly.sort_values(['opponent_team', 'week'])
            def_weekly[f'def_allowed_{stat}'] = def_weekly.groupby('opponent_team')[stat].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            df = df.merge(def_weekly[['opponent_team', 'week', f'def_allowed_{stat}']], 
                          on=['opponent_team', 'week'], how='left')
            # Fill NaNs
            if f'def_allowed_{stat}' in df.columns:
                df[f'def_allowed_{stat}'] = df[f'def_allowed_{stat}'].fillna(df[stat].mean())

    # Fatigue & Reliability proxies
    df['fatigue_index'] = 0 # Simplified
    df['reliability'] = 0.5
    if 'location' in df.columns:
        df['is_home'] = (df['location'] == 'HOME').astype(int)
    else:
        df['is_home'] = 0.5

    # Filter to latest week
    latest_week = df['week'].max()
    week_df = df[df['week'] == latest_week].copy()
    print(f"🔮 Predicting for Week {latest_week} ({len(week_df)} players)...")
    
    predictions = []
    
    for stat, label in stats_map.items():
        model_path = MODELS_DIR / f"nfl/models/props/xgb_nfl_{stat}_v1.json"
        if not model_path.exists(): continue
        
        print(f"   Loading model: {model_path.name}")
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        
        features = [
            f'{stat}_last_1', f'{stat}_last_3', f'{stat}_last_5',
            f'{stat}_season_avg',
            f'{stat}_momentum',
            f'{stat}_stability',
            f'def_allowed_{stat}',
            'fatigue_index', 'reliability',
            'is_home'
        ]
        
        # Check if features exist (some def_allowed might be missing if merger failed)
        missing = [f for f in features if f not in week_df.columns]
        if missing:
            # Fill missing with 0 for safety
            for m in missing: week_df[m] = 0
            
        valid_rows = week_df.dropna(subset=features)
        if len(valid_rows) == 0: continue
        
        # Filter purely low-stat players (noise reduction)
        valid_rows = valid_rows[valid_rows[f'{stat}_season_avg'] > (10 if 'yards' in stat else 0.5)]

        preds = model.predict(valid_rows[features])
        
        for idx, (i, row) in enumerate(valid_rows.iterrows()):
            projection = preds[idx]
            line = row[f'{stat}_season_avg']
            
            # Smart Line
            line = round(line)
            if line == 0: line = 0.5
            
            # Pick & Confidence
            gap_val = abs(projection - line)
            if projection > line:
                pick = "OVER"
            else:
                pick = "UNDER"
                
            # Dynamic Confidence Calculation (V6 NFL)
            base_acc = 0.55
            if 'yards' in label.lower(): base_acc = 0.58
            
            gap_pct = gap_val / max(line, 0.1)
            
            # Heavy dampening for NFL low stats
            if line < 3.0: gap_pct *= 0.3
            elif line < 20.0: gap_pct *= 0.7
            
            # Less aggressive scaling for NFL variance
            conf = base_acc + (gap_pct * 0.35)
            
            conf = min(max(conf, 0.52), 0.82)

            # Data Cleaning
            pos = row.get('position')
            if not pos: pos = 'Flex'

            predictions.append({
                "player": row.get('player_display_name', row.get('player_name', 'Unknown')),
                "team": row.get('recent_team', 'NFL'),
                "prop": label,
                "line": float(line),
                "pick": pick,
                "confidence": float(round(conf, 2)),
                "trend": "🔥 Hot" if row.get(f'{stat}_momentum', 0) > 5 else "❄️ Cold" if row.get(f'{stat}_momentum', 0) < -5 else "➡️ Neutral",
                "player_avg": float(round(row[f'{stat}_season_avg'], 1)),
                "event_group": "Week " + str(latest_week),
                "matchup": f"{row.get('recent_team')} vs {row.get('opponent_team')}",
                "sport": "nfl",
                "position": pos
            })
            
    return predictions

def sanitize_for_json(obj):
    """Recursively clean NaN/Infinity for JSON compliance."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None # JSON null
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return sanitize_for_json(float(obj))
    return obj

def main():
    all_preds = {"sports": {}}
    
    # NBA
    nba_df = load_nba_data()
    if nba_df is not None:
        # Fill NaNs in dataframe first to avoid most issues
        nba_df = nba_df.where(pd.notnull(nba_df), None)
        nba_preds = generate_nba_predictions(nba_df)
        all_preds["sports"]["nba"] = {"predictions": nba_preds}
        print(f"✅ Generated {len(nba_preds)} NBA predictions")
        
    # NFL
    nfl_df = load_nfl_data()
    if nfl_df is not None:
        # Fill NaNs
        nfl_df = nfl_df.where(pd.notnull(nfl_df), None)
        nfl_preds = generate_nfl_predictions(nfl_df)
        all_preds["sports"]["nfl"] = {"predictions": nfl_preds}
        print(f"✅ Generated {len(nfl_preds)} NFL predictions")
        
    # Sanitize entire structure
    all_preds = sanitize_for_json(all_preds)
        
    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_preds, f, indent=2)
    print(f"\n💾 Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
