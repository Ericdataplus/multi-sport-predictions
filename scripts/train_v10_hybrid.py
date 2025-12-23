"""
V10 Hybrid Ensemble Model for NBA  
====================================
Combines the BEST of all approaches:

1. XGBoost + LightGBM (our proven V6 foundation)
2. Extended rolling windows (like 72.35% paper - use 40 games not 10)
3. Momentum features (multiple timeframes)
4. Market-aware features (simulate betting line info)
5. Stacked generalization (level 2 meta-learner)

Key insight from research: XGBoost outperforms deep learning on tabular data.
So we enhance XGBoost with better features rather than replacing it.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "nba"
MODELS_DIR = BASE_DIR / "models"


def load_data():
    """Load NBA data."""
    print("\n📊 Loading NBA data...")
    
    df = pd.read_csv(DATA_DIR / "games.csv")
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['game_date_est'])
    df = df.dropna(subset=['pts_home', 'pts_away'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Filter to recent seasons (like paper - focus on 2015+)
    df = df[df['date'].dt.year >= 2015]
    
    print(f"  Loaded {len(df)} games (2015+)")
    return df


def calculate_team_stats(df, team_id, idx, windows=[5, 10, 20, 40]):
    """
    Calculate multi-window rolling stats.
    Key insight from 72.35% paper: longer windows capture season trends.
    """
    prev = df.iloc[:idx]
    home_mask = prev['home_team_id'] == team_id
    away_mask = prev['visitor_team_id'] == team_id
    team_games = prev[home_mask | away_mask]
    
    if len(team_games) < 5:
        return None
    
    stats = {}
    
    # Get all stats for calculations
    all_pts = []
    all_pts_against = []
    all_wins = []
    all_fg = []
    all_fg3 = []
    all_reb = []
    all_ast = []
    
    for _, g in team_games.iterrows():
        if g['home_team_id'] == team_id:
            all_pts.append(g['pts_home'])
            all_pts_against.append(g['pts_away'])
            all_wins.append(1 if g['pts_home'] > g['pts_away'] else 0)
            all_fg.append(g.get('fg_pct_home', 0.45))
            all_fg3.append(g.get('fg3_pct_home', 0.35))
            all_reb.append(g.get('reb_home', 42))
            all_ast.append(g.get('ast_home', 22))
        else:
            all_pts.append(g['pts_away'])
            all_pts_against.append(g['pts_home'])
            all_wins.append(1 if g['pts_away'] > g['pts_home'] else 0)
            all_fg.append(g.get('fg_pct_away', 0.45))
            all_fg3.append(g.get('fg3_pct_away', 0.35))
            all_reb.append(g.get('reb_away', 42))
            all_ast.append(g.get('ast_away', 22))
    
    # Calculate stats for each window
    for w in windows:
        suffix = f'_{w}'
        n = min(len(all_pts), w)
        recent = slice(-n, None)
        
        pts = all_pts[recent]
        pts_a = all_pts_against[recent]
        wins = all_wins[recent]
        fg = [x for x in all_fg[recent] if x > 0]
        fg3 = [x for x in all_fg3[recent] if x > 0]
        reb = all_reb[recent]
        ast = all_ast[recent]
        
        stats[f'win_pct{suffix}'] = np.mean(wins) if wins else 0.5
        stats[f'pts{suffix}'] = np.mean(pts) if pts else 100
        stats[f'pts_against{suffix}'] = np.mean(pts_a) if pts_a else 100
        stats[f'net{suffix}'] = stats[f'pts{suffix}'] - stats[f'pts_against{suffix}']
        stats[f'fg_pct{suffix}'] = np.mean(fg) if fg else 0.45
        stats[f'fg3_pct{suffix}'] = np.mean(fg3) if fg3 else 0.35
        stats[f'reb{suffix}'] = np.mean(reb) if reb else 42
        stats[f'ast{suffix}'] = np.mean(ast) if ast else 22
        stats[f'pts_std{suffix}'] = np.std(pts) if len(pts) > 1 else 10
    
    # Trend features (comparing windows - captures momentum)
    stats['momentum_short'] = stats['win_pct_5'] - stats['win_pct_20']  # Recent vs medium
    stats['momentum_long'] = stats['win_pct_10'] - stats['win_pct_40']  # Medium vs season
    stats['scoring_trend'] = stats['pts_5'] - stats['pts_20']
    stats['defense_trend'] = stats['pts_against_20'] - stats['pts_against_5']  # Better defense = positive
    
    # Consistency (lower std = more predictable)
    stats['consistency'] = 1 / (1 + stats['pts_std_20'])
    
    # Games played (sample size indicator)
    stats['games'] = len(all_pts)
    
    return stats


def create_features(df):
    """Create enhanced feature set."""
    print("\n🔧 Creating V10 enhanced features...")
    
    features = []
    targets_ml = []
    targets_spread = []
    targets_total = []
    
    for idx in range(len(df)):
        if idx < 200:  # Need more history for 40-game window
            continue
        
        row = df.iloc[idx]
        home_id = row['home_team_id']
        away_id = row['visitor_team_id']
        
        home = calculate_team_stats(df, home_id, idx)
        away = calculate_team_stats(df, away_id, idx)
        
        if home is None or away is None:
            continue
        
        # Feature engineering
        f = {}
        
        # 1. Differential features for each window (most important)
        for w in [5, 10, 20, 40]:
            f[f'win_pct_diff_{w}'] = home[f'win_pct_{w}'] - away[f'win_pct_{w}']
            f[f'net_diff_{w}'] = home[f'net_{w}'] - away[f'net_{w}']
            f[f'pts_diff_{w}'] = home[f'pts_{w}'] - away[f'pts_{w}']
            f[f'fg_diff_{w}'] = home[f'fg_pct_{w}'] - away[f'fg_pct_{w}']
        
        # 2. Momentum differential (key predictor)
        f['momentum_short_diff'] = home['momentum_short'] - away['momentum_short']
        f['momentum_long_diff'] = home['momentum_long'] - away['momentum_long']
        f['scoring_trend_diff'] = home['scoring_trend'] - away['scoring_trend']
        f['defense_trend_diff'] = home['defense_trend'] - away['defense_trend']
        
        # 3. Raw stats (helps with absolute strength)
        f['home_win_pct_20'] = home['win_pct_20']
        f['away_win_pct_20'] = away['win_pct_20']
        f['home_net_20'] = home['net_20']
        f['away_net_20'] = away['net_20']
        
        # 4. Matchup features
        f['total_pace'] = (home['pts_20'] + away['pts_20']) / 200
        f['mismatch'] = abs(home['net_20'] - away['net_20'])
        f['consistency_diff'] = home['consistency'] - away['consistency']
        f['sample_size'] = min(home['games'], away['games']) / 40
        
        # 5. Simulated market features
        # (In production, we'd use real betting lines)
        implied_spread = (home['net_20'] - away['net_20']) * 0.4
        f['implied_spread'] = implied_spread
        f['home_favored'] = 1 if implied_spread > 0 else 0
        implied_total = (home['pts_20'] + away['pts_20']) * 0.98
        f['implied_total'] = implied_total
        
        features.append(f)
        
        # Targets
        home_pts = row['pts_home']
        away_pts = row['pts_away']
        
        targets_ml.append(1.0 if home_pts > away_pts else 0.0)
        targets_spread.append(1.0 if (home_pts - away_pts) > implied_spread else 0.0)
        targets_total.append(1.0 if (home_pts + away_pts) > implied_total else 0.0)
        
        if len(features) % 2000 == 0:
            print(f"    Processed {len(features)} games...")
    
    X = pd.DataFrame(features)
    print(f"\n  Created {len(X)} samples with {len(X.columns)} features")
    
    return X, np.array(targets_ml), np.array(targets_spread), np.array(targets_total)


def train_stacked_model(X, y, bet_type='moneyline'):
    """
    Train stacked ensemble:
    Level 1: XGBoost + LightGBM + ExtraTrees
    Level 2: Logistic Regression meta-learner
    """
    print(f"\n  Training {bet_type.upper()}...")
    
    # Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time-based split
    split = int(len(X) * 0.85)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Level 1 models
    models = {}
    oof_preds = np.zeros((len(X_train), 3))  # Out-of-fold predictions
    test_preds = np.zeros((len(X_test), 3))
    
    # 5-fold time series CV for stacking
    tscv = TimeSeriesSplit(n_splits=5)
    
    # XGBoost (optimized params)
    print("    Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.02,
        reg_lambda=5.0, reg_alpha=1.0,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=3,
        random_state=42, n_jobs=-1
    )
    
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        xgb_model.fit(X_train[tr_idx], y_train[tr_idx], 
                     eval_set=[(X_train[val_idx], y_train[val_idx])],
                     verbose=False)
        oof_preds[val_idx, 0] = xgb_model.predict_proba(X_train[val_idx])[:, 1]
    
    xgb_model.fit(X_train, y_train, verbose=False)
    test_preds[:, 0] = xgb_model.predict_proba(X_test)[:, 1]
    models['xgb'] = xgb_model
    
    # LightGBM
    print("    Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.02,
        reg_lambda=5.0, reg_alpha=1.0,
        subsample=0.8, colsample_bytree=0.7,
        min_child_samples=10,
        random_state=42, n_jobs=-1, verbose=-1
    )
    
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        lgb_model.fit(X_train[tr_idx], y_train[tr_idx],
                     eval_set=[(X_train[val_idx], y_train[val_idx])])
        oof_preds[val_idx, 1] = lgb_model.predict_proba(X_train[val_idx])[:, 1]
    
    lgb_model.fit(X_train, y_train)
    test_preds[:, 1] = lgb_model.predict_proba(X_test)[:, 1]
    models['lgb'] = lgb_model
    
    # Gradient Boosting (different perspective)
    print("    Training GradientBoosting...")
    from sklearn.ensemble import GradientBoostingClassifier
    gb_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        gb_model.fit(X_train[tr_idx], y_train[tr_idx])
        oof_preds[val_idx, 2] = gb_model.predict_proba(X_train[val_idx])[:, 1]
    
    gb_model.fit(X_train, y_train)
    test_preds[:, 2] = gb_model.predict_proba(X_test)[:, 1]
    models['gb'] = gb_model
    
    # Level 2: Meta-learner (stacking)
    print("    Training meta-learner (stacking)...")
    
    # Only use non-zero rows (from CV folds)
    valid_mask = oof_preds[:, 0] > 0
    meta_train = oof_preds[valid_mask]
    meta_y = y_train[valid_mask]
    
    meta_model = LogisticRegression(C=1.0, random_state=42)
    meta_model.fit(meta_train, meta_y)
    
    # Final predictions
    final_preds = meta_model.predict_proba(test_preds)[:, 1]
    
    # Calibration
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(final_preds, y_test)
    calibrated_preds = calibrator.predict(final_preds)
    
    # Evaluate
    pred_binary = (final_preds > 0.5).astype(int)
    acc = accuracy_score(y_test, pred_binary)
    auc = roc_auc_score(y_test, final_preds)
    
    print(f"    ✅ Accuracy: {acc:.1%}, AUC: {auc:.4f}")
    
    # Individual model contributions
    xgb_acc = accuracy_score(y_test, (test_preds[:, 0] > 0.5).astype(int))
    lgb_acc = accuracy_score(y_test, (test_preds[:, 1] > 0.5).astype(int))
    gb_acc = accuracy_score(y_test, (test_preds[:, 2] > 0.5).astype(int))
    print(f"       XGB: {xgb_acc:.1%}, LGB: {lgb_acc:.1%}, GB: {gb_acc:.1%}")
    
    return {
        'models': models,
        'meta_model': meta_model,
        'scaler': scaler,
        'calibrator': calibrator,
        'accuracy': acc,
        'auc': auc,
        'features': list(X.columns)
    }


def main():
    """Train V10 model."""
    print("\n" + "="*60)
    print("🚀 V10 HYBRID ENSEMBLE MODEL")
    print("="*60)
    print("XGBoost + LightGBM + GB with multi-window features & stacking")
    
    # Load data
    df = load_data()
    
    # Create features
    X, y_ml, y_spread, y_total = create_features(df)
    
    # Train models
    results = {}
    
    ml_result = train_stacked_model(X, y_ml, 'moneyline')
    results['moneyline'] = ml_result
    
    spread_result = train_stacked_model(X, y_spread, 'spread')
    results['spread'] = spread_result
    
    total_result = train_stacked_model(X, y_total, 'total')
    results['total'] = total_result
    
    # Summary
    print("\n" + "="*60)
    print("📊 V10 vs V6 COMPARISON")
    print("="*60)
    v6 = {'moneyline': 0.654, 'spread': 0.734, 'total': 0.55}
    
    for bt in ['moneyline', 'spread', 'total']:
        v6_acc = v6[bt]
        v10_acc = results[bt]['accuracy']
        diff = (v10_acc - v6_acc) * 100
        symbol = '↑' if diff > 0 else '↓'
        print(f"  {bt.upper():10} | V6: {v6_acc:.1%} → V10: {v10_acc:.1%} | {symbol}{abs(diff):.1f}pp")
    
    # Save
    save_dict = {bt: {
        'accuracy': r['accuracy'],
        'auc': r['auc'],
        'features': r['features']
    } for bt, r in results.items()}
    
    with open(MODELS_DIR / "v10_nba_metrics.json", 'w') as f:
        json.dump(save_dict, f, indent=2)
    
    # Save full model
    with open(MODELS_DIR / "v10_nba_stacked.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n  Saved to: {MODELS_DIR / 'v10_nba_stacked.pkl'}")
    
    return results


if __name__ == "__main__":
    results = main()
