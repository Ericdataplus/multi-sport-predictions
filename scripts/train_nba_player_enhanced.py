
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Config
DATA_DIR = Path("data/nba")
GAMES_PATH = DATA_DIR / "games.csv"
DETAILS_PATH = DATA_DIR / "games_details.csv"

def load_and_prep_data():
    print("Loading data...")
    if not GAMES_PATH.exists() or not DETAILS_PATH.exists():
        print("Data files not found in data/nba/")
        return None, None

    # Load Games (Target info)
    games = pd.read_csv(GAMES_PATH)
    games = games.drop_duplicates(subset=['GAME_ID'])
    games['GAME_DATE_EST'] = pd.to_datetime(games['GAME_DATE_EST'])
    games = games.sort_values('GAME_DATE_EST')
    
    # Simple Target: Home Team Wins
    games['target'] = games['HOME_TEAM_WINS']
    
    # Load Player Details
    details = pd.read_csv(DETAILS_PATH)
    
    # Merge Date onto Details for rolling calcs
    date_map = games.set_index('GAME_ID')['GAME_DATE_EST']
    details['GAME_DATE'] = details['GAME_ID'].map(date_map)
    details = details.dropna(subset=['GAME_DATE']) # Drop games not in games.csv
    
    print(f"Loaded {len(games)} games and {len(details)} player records.")
    return games, details

def calculate_player_rolling_stats(details):
    print("Calculating player rolling stats (this may take a moment)...")
    
    # Sort for rolling calcs
    details = details.sort_values(['PLAYER_ID', 'GAME_DATE'])
    
    # Metrics to track
    metrics = ['PTS', 'AST', 'REB', 'PLUS_MINUS']
    
    # Calculate rolling averages per player using TRANSFORM to preserve index alignment
    for col in metrics:
        # We must SHIFT(1) so we don't use current game stats (LEAKAGE)
        # transform returns aligned series
        details[f'{col}_R10'] = details.groupby('PLAYER_ID')[col].transform(
            lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
        )
    
    # Fill remaining NaNs (e.g. first game of career) with 0 or season avg?
    # 0 is safe for 'sum' aggregations later.
    cols_to_fill = [f'{c}_R10' for c in metrics]
    details[cols_to_fill] = details[cols_to_fill].fillna(0)
    
    return details

def aggregate_team_features(details):
    print("Aggregating player stats into team features...")
    
    features = ['PTS_R10', 'AST_R10', 'REB_R10', 'PLUS_MINUS_R10']
    
    # Group by Game and Team
    # We want to represent the "Team Capacity" based on players who played.
    team_agg = details.groupby(['GAME_ID', 'TEAM_ID'])[features].agg(['sum', 'mean', 'max'])
    
    # Flatten columns
    team_agg.columns = [f"TEAM_PL_{x}_{y}" for x, y in team_agg.columns]
    team_agg = team_agg.reset_index()
    
    return team_agg

def merge_features_to_games(games, team_agg):
    print("Merging features to game records...")
    
    # Merge Home Team Features
    games = games.merge(
        team_agg, 
        left_on=['GAME_ID', 'HOME_TEAM_ID'], 
        right_on=['GAME_ID', 'TEAM_ID'],
        suffixes=('', '_HOME')
    )
    # Rename columns to identify as HOME
    rename_dict_home = {c: f"HOME_{c}" for c in team_agg.columns if c not in ['GAME_ID', 'TEAM_ID']}
    games = games.rename(columns=rename_dict_home)
    
    # Merge Away Team Features
    games = games.merge(
        team_agg, 
        left_on=['GAME_ID', 'VISITOR_TEAM_ID'], 
        right_on=['GAME_ID', 'TEAM_ID'], 
        suffixes=('', '_AWAY')
    )
    # Rename columns to identify as AWAY
    rename_dict_away = {c: f"AWAY_{c}" for c in team_agg.columns if c not in ['GAME_ID', 'TEAM_ID']}
    games = games.rename(columns=rename_dict_away)
    
    return games

def train_and_evaluate(df):
    # Select Features
    # We want both Player Aggregates (TEAM_PL_...) and Team Macro Stats (TEAM_..._R10)
    # The suffix keys defined earlier created columns like HOME_TEAM_PTS_R10
    
    feature_cols = [c for c in df.columns if 'TEAM_' in c and ('_R10' in c or 'TEAM_PL_' in c)]
    # Filter out ID columns just in case
    feature_cols = [c for c in df.columns if c not in ['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'TEAM_ID', 'target', 'GAME_DATE_EST', 'GAME_STATUS_TEXT', 'SEASON']]
    # Actually, let's be specific to avoid "HOME_TEAM_WINS" leakage which is in games.csv originally!
    
    # Safe list of feature patterns
    feature_cols = [c for c in df.columns if 
                    (c.startswith('HOME_') or c.startswith('AWAY_')) and 
                    c not in ['HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_TEAM_WINS'] and
                    df[c].dtype in ['float64', 'int64', 'int32', 'float32']]
    
    # --- "SMART" FEATURE ENGINEERING: DIFFERENTIALS ---
    print("Engineering Differential (Matchup) Features...")
    # Trees struggle to learn "A - B" easily. We help them by providing the difference explicitly.
    
    # Identify matching Home/Away pairs
    home_cols = [c for c in feature_cols if c.startswith('HOME_')]
    for h_col in home_cols:
        base_name = h_col.replace('HOME_', '')
        a_col = f"AWAY_{base_name}"
        if a_col in df.columns:
            # Create the Differential Feature
            # Ensure no NaNs before subtraction to avoid exploding Errors if indices weird
            df[f'DIFF_{base_name}'] = df[h_col].fillna(0) - df[a_col].fillna(0)
            
    # --- "LEAN & MEAN" FEATURE SELECTION (UPDATED) ---
    print("Selecting Feature Set (Differentials + Mad Scientist Features)...")
    
    # Keeping:
    # 1. Player Differentials (DIFF_)
    # 2. Mad ELO Features (HOME_ELO_MAD, AWAY_ELO_MAD, DIFF_ELO_MAD, ELO_WIN_PROB)
    # 3. Context (REST_DAYS)
    
    final_features = [c for c in df.columns if c.startswith('DIFF_')]
    
    # Add Mad ELO specifics
    elo_cols = [c for c in df.columns if '_ELO_MAD' in c or 'ELO_WIN_PROB' in c]
    final_features.extend(elo_cols)
    
    # Add Rest Days
    # Check for REST_DAYS existence (handle columns dynamically)
    rest_cols = [c for c in df.columns if 'REST_DAYS' in c]
    final_features.extend(rest_cols)
    
    print(f"Debug: Selected {len(final_features)} features.")
    print(f"Debug: Feature list: {final_features}")
    
    feature_cols = final_features
    
    target_col = 'target'
    
    # Prepare Data
    df[feature_cols] = df[feature_cols].fillna(0)
    X = df[feature_cols]
    y = df[target_col]
    
    # Time-based split
    split_idx = int(len(df) * 0.85) # Validation on last 15%
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTraining Hybrid ENSEMBLE on {len(X_train)} games, testing on {len(X_test)}...")
    
    # --- "MODERN" TABULAR APPROACH: ENSEMBLING ---
    # Combining XGBoost (Depth-wise) and LightGBM (Leaf-wise) often yields better generalization.
    
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=600,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        eval_metric='logloss',
        early_stopping_rounds=30,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    print("Training LightGBM...")
    import lightgbm as lgb
    lgb_model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='logloss', callbacks=[lgb.early_stopping(30, verbose=False)])
    
    # Ensemble Prediction (Soft Voting / Averaging probabilities)
    print("Calculating Ensemble Predictions...")
    xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
    lgb_preds = lgb_model.predict_proba(X_test)[:, 1]
    
    # Simple Average (often beats complex weighting)
    final_probs = (xgb_preds + lgb_preds) / 2
    final_preds = (final_probs > 0.5).astype(int)
    
    acc = accuracy_score(y_test, final_preds)
    
    print(f"\n{'='*50}")
    print(f"SMART ENSEMBLE MODEL ACCURACY: {acc:.2%}")
    print(f"{'='*50}")
    
    # Check individual performance
    xgb_acc = accuracy_score(y_test, (xgb_preds > 0.5).astype(int))
    lgb_acc = accuracy_score(y_test, (lgb_preds > 0.5).astype(int))
    print(f"XGBoost Alone: {xgb_acc:.2%}")
    print(f"LightGBM Alone: {lgb_acc:.2%}")
    
    # Feature Importance (from XGB)
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features (XGB):")
    print(importance.head(5))
    
    return acc

class MadScientistElo:
    def __init__(self, k_factor=20, margin_multiplier=True):
        self.team_ratings = {} # dict of team_id -> rating
        self.k = k_factor
        self.use_margin = margin_multiplier
        self.base_rating = 1500
        
    def get_rating(self, team_id):
        return self.team_ratings.get(team_id, self.base_rating)
    
    def update_game(self, home_id, away_id, home_won, margin, is_neutral=False):
        # 1. Get current ratings
        r_home = self.get_rating(home_id)
        r_away = self.get_rating(away_id)
        
        # 2. Calculate Expected Outcome (Logistic Curve)
        # Home court advantage usually ~60-80 Elo points
        h_adv = 70 if not is_neutral else 0
        dr = (r_home + h_adv) - r_away
        e_home = 1 / (1 + 10 ** (-dr / 400))
        
        # 3. Calculate Multiplier (Mad Scientist Part)
        # Margin of Victory Multiplier = ln(abs(PD) + 1) * 2.2 / ((EloDiff)*.001 + 2.2)
        # Simplified: a log scale of margin
        k_mult = 1.0
        if self.use_margin:
            k_mult = np.log(abs(margin) + 1) * 0.5
            
        # 4. Update Ratings
        actual = 1.0 if home_won else 0.0
        delta = self.k * k_mult * (actual - e_home)
        
        self.team_ratings[home_id] = r_home + delta
        self.team_ratings[away_id] = r_away - delta
        
        return r_home, r_away, e_home # Return stats BEFORE update (Forecast)

def calculate_mad_elo_features(games):
    print("Initializing MAD SCIENTIST ELO Engine...")
    
    # Sort chronologically or Elo breaks
    games = games.sort_values('GAME_DATE_EST')
    
    # Initialize Tracker
    elo = MadScientistElo(k_factor=30, margin_multiplier=True)
    
    # Storage for features
    home_elos = []
    away_elos = []
    elo_probs = []
    
    # Iterate and simulate the season
    # We must loop row-by-row to prevent looking into the future
    for idx, row in games.iterrows():
        hid = row['HOME_TEAM_ID']
        aid = row['VISITOR_TEAM_ID']
        
        # 1. Get Ratings (Forecast) - BEFORE the game result is known
        h_rating, a_rating, win_prob = elo.update_game(
            hid, 
            aid, 
            row['HOME_TEAM_WINS'] == 1,
            row['PTS_home'] - row['PTS_away']
        )
        
        home_elos.append(h_rating)
        away_elos.append(a_rating)
        elo_probs.append(win_prob)
        
    # Add to DataFrame
    games['HOME_ELO_MAD'] = home_elos
    games['AWAY_ELO_MAD'] = away_elos
    games['ELO_WIN_PROB'] = elo_probs
    games['DIFF_ELO_MAD'] = games['HOME_ELO_MAD'] - games['AWAY_ELO_MAD']
    
    print("  Elo simulation complete.")
    return games[['GAME_ID', 'HOME_ELO_MAD', 'AWAY_ELO_MAD', 'DIFF_ELO_MAD', 'ELO_WIN_PROB']]

def calculate_team_rolling_stats(games):
    # Just return basic rest stats, we rely on ELO for form now
    print("Calculating Basic Context Stats...")
    
    home = games[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID']].copy()
    home.columns = ['GAME_ID', 'GAME_DATE', 'TEAM_ID']
    home['IS_HOME'] = 1
    
    away = games[['GAME_ID', 'GAME_DATE_EST', 'VISITOR_TEAM_ID']].copy()
    away.columns = ['GAME_ID', 'GAME_DATE', 'TEAM_ID'] 
    
    team_games = pd.concat([home, away]).sort_values(['TEAM_ID', 'GAME_DATE'])
    
    team_games['PREV_GAME_DATE'] = team_games.groupby('TEAM_ID')['GAME_DATE'].shift(1)
    team_games['REST_DAYS'] = (team_games['GAME_DATE'] - team_games['PREV_GAME_DATE']).dt.days
    team_games['REST_DAYS'] = team_games['REST_DAYS'].fillna(3).clip(upper=7)
    
    df_features = team_games[['GAME_ID', 'TEAM_ID', 'REST_DAYS']].copy()
    
    return df_features

def main():
    print("Starting Training Pipeline for 'Smart' Hybrid Model...")
    
    # 1. Load
    games, details = load_and_prep_data()
    if games is None: return
    
    # 2. Player Features
    details = calculate_player_rolling_stats(details)
    player_agg = aggregate_team_features(details)
    # 3. Team Features (Rest + Mad Scientist ELO)
    context_stats = calculate_team_rolling_stats(games)
    elo_stats = calculate_mad_elo_features(games)
    
    # 4. Merge Everything
    print("Merging all datasets...")
    
    games_enriched = merge_features_to_games(games, player_agg)
    
    # Merge Elo Stats (Already one row per game with HOME/AWAY columns)
    games_enriched = games_enriched.merge(elo_stats, on='GAME_ID')
    
    # Merge Context (Context stats only has GAME_ID, TEAM_ID, REST_DAYS)
    # Strategy: Filter context_stats for Home and Away ID sets, Rename UP FRONT, then merge.
    
    # 1. Prepare Home Context
    context_home = context_stats.copy()
    context_home = context_home.rename(columns={'TEAM_ID': 'HOME_TEAM_ID', 'REST_DAYS': 'HOME_REST_DAYS'})
    # Join on GAME_ID and HOME_TEAM_ID
    games_enriched = games_enriched.merge(context_home, on=['GAME_ID', 'HOME_TEAM_ID'], how='left')
    
    # 2. Prepare Away Context
    context_away = context_stats.copy()
    context_away = context_away.rename(columns={'TEAM_ID': 'VISITOR_TEAM_ID', 'REST_DAYS': 'AWAY_REST_DAYS'})
    # Join on GAME_ID and VISITOR_TEAM_ID
    games_enriched = games_enriched.merge(context_away, on=['GAME_ID', 'VISITOR_TEAM_ID'], how='left')
    
    # Clean up duplicate columns if any (Defensive programming)
    games_enriched = games_enriched.loc[:, ~games_enriched.columns.duplicated()]
    
    # 5. Train
    train_and_evaluate(games_enriched)

if __name__ == "__main__":
    main()
