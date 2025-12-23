"""
V8 Random Forest + Deep Features Model
=======================================
Trying to replicate ~80% accuracy from research papers.
Uses more advanced features and Random Forest.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "nba"
MODELS_DIR = BASE_DIR / "models"


def load_nba_data():
    """Load NBA data."""
    print("  Loading NBA data...")
    
    df = pd.read_csv(DATA_DIR / "games.csv")
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['game_date_est'])
    df = df.dropna(subset=['pts_home', 'pts_away'])
    df = df[df['date'].dt.year >= 2015]  # Recent data only
    df['season'] = df['date'].apply(lambda x: x.year if x.month >= 10 else x.year - 1)
    
    print(f"  Loaded {len(df)} games")
    return df.sort_values('date').reset_index(drop=True)


def calculate_advanced_features(df, team_id, games_before, season):
    """Calculate advanced features like in research papers."""
    season_games = games_before[games_before['season'] == season]
    
    if len(season_games) < 5:
        return None
    
    # Find games where this team played
    home_games = season_games[season_games['home_team_id'] == team_id]
    away_games = season_games[season_games['visitor_team_id'] == team_id]
    
    if len(home_games) + len(away_games) < 3:
        return None
    
    # Aggregate stats
    pts_list = []
    pts_against_list = []
    fg_pct_list = []
    fg3_pct_list = []
    ft_pct_list = []
    reb_list = []
    ast_list = []
    wins = 0
    
    for _, g in home_games.iterrows():
        pts_list.append(g['pts_home'])
        pts_against_list.append(g['pts_away'])
        if g.get('fg_pct_home', 0) > 0:
            fg_pct_list.append(g['fg_pct_home'])
            fg3_pct_list.append(g.get('fg3_pct_home', 0.35))
            ft_pct_list.append(g.get('ft_pct_home', 0.75))
            reb_list.append(g.get('reb_home', 42))
            ast_list.append(g.get('ast_home', 22))
        if g['pts_home'] > g['pts_away']:
            wins += 1
    
    for _, g in away_games.iterrows():
        pts_list.append(g['pts_away'])
        pts_against_list.append(g['pts_home'])
        if g.get('fg_pct_away', 0) > 0:
            fg_pct_list.append(g['fg_pct_away'])
            fg3_pct_list.append(g.get('fg3_pct_away', 0.35))
            ft_pct_list.append(g.get('ft_pct_away', 0.75))
            reb_list.append(g.get('reb_away', 42))
            ast_list.append(g.get('ast_away', 22))
        if g['pts_away'] > g['pts_home']:
            wins += 1
    
    if not pts_list:
        return None
    
    n = len(pts_list)
    
    # Basic stats
    stats = {
        'pts_mean': np.mean(pts_list),
        'pts_std': np.std(pts_list),
        'pts_against_mean': np.mean(pts_against_list),
        'net_rating': np.mean(pts_list) - np.mean(pts_against_list),
        'win_pct': wins / n,
        'fg_pct': np.mean(fg_pct_list) if fg_pct_list else 0.45,
        'fg3_pct': np.mean(fg3_pct_list) if fg3_pct_list else 0.35,
        'ft_pct': np.mean(ft_pct_list) if ft_pct_list else 0.75,
        'reb': np.mean(reb_list) if reb_list else 42,
        'ast': np.mean(ast_list) if ast_list else 22,
        'games': n,
    }
    
    # Advanced metrics (like papers use)
    stats['off_rating'] = stats['pts_mean']
    stats['def_rating'] = stats['pts_against_mean']
    stats['efg_pct'] = stats['fg_pct'] * 1.1  # Approximate eFG%
    stats['ts_pct'] = stats['pts_mean'] / (2 * (stats['fg_pct'] * 80 + 0.44 * stats['ft_pct'] * 20))  # Approx TS%
    
    # Momentum (last 5 games)
    last5_pts = pts_list[-5:] if len(pts_list) >= 5 else pts_list
    last5_against = pts_against_list[-5:] if len(pts_against_list) >= 5 else pts_against_list
    stats['momentum'] = np.mean(last5_pts) - np.mean(last5_against)
    
    # Consistency (low std = more consistent)
    stats['consistency'] = 1 / (1 + stats['pts_std'])
    
    # Home/Away splits
    if len(home_games) > 0 and len(away_games) > 0:
        home_pts = home_games['pts_home'].mean()
        away_pts = away_games['pts_away'].mean()
        stats['home_away_diff'] = home_pts - away_pts
    else:
        stats['home_away_diff'] = 0
    
    return stats


def create_features(df):
    """Create training data."""
    print("  Creating features...")
    
    features = []
    targets = []
    
    for idx, row in df.iterrows():
        date = row['date']
        season = row['season']
        home_id = row['home_team_id']
        away_id = row['visitor_team_id']
        
        games_before = df[df['date'] < date]
        
        home_stats = calculate_advanced_features(df, home_id, games_before, season)
        away_stats = calculate_advanced_features(df, away_id, games_before, season)
        
        if home_stats is None or away_stats is None:
            continue
        
        # Feature differences (like papers do)
        f = {
            'win_pct_diff': home_stats['win_pct'] - away_stats['win_pct'],
            'pts_diff': home_stats['pts_mean'] - away_stats['pts_mean'],
            'pts_against_diff': away_stats['pts_against_mean'] - home_stats['pts_against_mean'],
            'net_rating_diff': home_stats['net_rating'] - away_stats['net_rating'],
            'fg_pct_diff': home_stats['fg_pct'] - away_stats['fg_pct'],
            'fg3_pct_diff': home_stats['fg3_pct'] - away_stats['fg3_pct'],
            'ft_pct_diff': home_stats['ft_pct'] - away_stats['ft_pct'],
            'reb_diff': home_stats['reb'] - away_stats['reb'],
            'ast_diff': home_stats['ast'] - away_stats['ast'],
            'momentum_diff': home_stats['momentum'] - away_stats['momentum'],
            'consistency_diff': home_stats['consistency'] - away_stats['consistency'],
            'off_rating_diff': home_stats['off_rating'] - away_stats['off_rating'],
            'def_rating_diff': away_stats['def_rating'] - home_stats['def_rating'],
            
            # Raw stats for context
            'home_win_pct': home_stats['win_pct'],
            'away_win_pct': away_stats['win_pct'],
            'home_net_rating': home_stats['net_rating'],
            'away_net_rating': away_stats['net_rating'],
            'home_games': home_stats['games'],
            'away_games': away_stats['games'],
        }
        
        features.append(f)
        targets.append(1 if row['pts_home'] > row['pts_away'] else 0)
        
        if len(features) % 2000 == 0:
            print(f"    Processed {len(features)} games...")
    
    print(f"  Created {len(features)} samples")
    return pd.DataFrame(features), np.array(targets)


def train_models(X, y):
    """Train multiple models and compare."""
    print("\n  Training models...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42
    )
    
    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = {}
    
    # 1. Random Forest (like papers)
    print("\n  1. Random Forest (max_depth=20, n=500)...")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=20, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    results['random_forest'] = {'acc': rf_acc, 'auc': rf_auc, 'model': rf}
    print(f"     → Accuracy: {rf_acc:.1%}, AUC: {rf_auc:.4f}")
    
    # 2. XGBoost with tuned params
    print("\n  2. XGBoost (tuned)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000, max_depth=6, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, reg_alpha=0.5,
        random_state=42, n_jobs=-1, use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
    results['xgboost'] = {'acc': xgb_acc, 'auc': xgb_auc, 'model': xgb_model}
    print(f"     → Accuracy: {xgb_acc:.1%}, AUC: {xgb_auc:.4f}")
    
    # 3. Gradient Boosting
    print("\n  3. Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        random_state=42
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])
    results['gradient_boost'] = {'acc': gb_acc, 'auc': gb_auc, 'model': gb}
    print(f"     → Accuracy: {gb_acc:.1%}, AUC: {gb_auc:.4f}")
    
    # 4. Ensemble of all three
    print("\n  4. Ensemble (RF + XGB + GB)...")
    rf_proba = rf.predict_proba(X_test)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    gb_proba = gb.predict_proba(X_test)[:, 1]
    ensemble_proba = (rf_proba + xgb_proba + gb_proba) / 3
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    ensemble_auc = roc_auc_score(y_test, ensemble_proba)
    results['ensemble'] = {'acc': ensemble_acc, 'auc': ensemble_auc}
    print(f"     → Accuracy: {ensemble_acc:.1%}, AUC: {ensemble_auc:.4f}")
    
    return results, scaler


def main():
    print("\n" + "="*60)
    print("V8 RANDOM FOREST + ADVANCED FEATURES MODEL")
    print("="*60)
    print("Attempting to replicate ~80% accuracy from research papers")
    
    df = load_nba_data()
    X, y = create_features(df)
    
    print(f"\n  Features: {X.shape}")
    
    results, scaler = train_models(X, y)
    
    print("\n" + "="*60)
    print("FINAL RESULTS - V8 NBA MONEYLINE")
    print("="*60)
    
    best_name = max(results.keys(), key=lambda k: results[k]['acc'])
    best_acc = results[best_name]['acc']
    
    for name, r in sorted(results.items(), key=lambda x: -x[1]['acc']):
        marker = " 🏆" if name == best_name else ""
        print(f"  {name:20} | {r['acc']:.1%} | AUC: {r['auc']:.4f}{marker}")
    
    print("\n  Comparison:")
    print(f"    V6 Moneyline: 65.4%")
    print(f"    V7 Moneyline: 65.2%")
    print(f"    V8 Best:      {best_acc:.1%}")
    
    diff = (best_acc - 0.654) * 100
    symbol = '↑' if diff > 0 else '↓'
    print(f"    Improvement:  {symbol}{abs(diff):.1f}pp vs V6")
    
    # Save best model
    best_model = results[best_name].get('model')
    if best_model:
        path = MODELS_DIR / "v8_nba_best.pkl"
        with open(path, 'wb') as f:
            pickle.dump({
                'model': best_model,
                'scaler': scaler,
                'features': list(X.columns),
                'accuracy': best_acc,
                'name': best_name
            }, f)
        print(f"\n  Saved: {path}")


if __name__ == "__main__":
    main()
