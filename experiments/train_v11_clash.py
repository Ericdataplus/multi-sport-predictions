"""
V11 Behavioral Clash Model for NBA
====================================
Core Innovation: Predict based on HOW behaviors interact in matchups.

The insight: Games are won in mismatches, not absolute strength.

Key Clash Features:
1. Steal Rate vs Turnover Rate = Defensive exploitation potential
2. Block Rate vs Shot Distribution = Paint protection matchup
3. Pace differential = Who controls tempo?
4. Assist Rate spread = Ball movement quality gap
5. Foul differential = Discipline vs aggression clash

Plus: Behavioral trends (is team improving or declining?)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "nba"
MODELS_DIR = BASE_DIR / "models"


def load_data():
    """Load NBA games with detailed stats."""
    print("\n📊 Loading NBA data...")
    
    # Main games file
    df = pd.read_csv(DATA_DIR / "games.csv")
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['game_date_est'])
    df = df.dropna(subset=['pts_home', 'pts_away'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Try to load detailed stats if available
    details_path = DATA_DIR / "games_details.csv"
    if details_path.exists():
        print("  Loading detailed player stats for behavioral aggregation...")
        details = pd.read_csv(details_path)
        details.columns = [c.lower() for c in details.columns]
        
        # Aggregate to game level
        game_stats = details.groupby('game_id').agg({
            'stl': 'sum',
            'blk': 'sum',
            'to': 'sum',
            'pf': 'sum',
            'ast': 'sum',
            'oreb': 'sum',
            'dreb': 'sum',
        }).reset_index() if 'stl' in details.columns else None
        
        if game_stats is not None:
            df = df.merge(game_stats, left_on='game_id', right_on='game_id', how='left')
    
    # Filter to recent seasons
    df = df[df['date'].dt.year >= 2015]
    
    print(f"  Loaded {len(df)} games")
    print(f"  Columns: {list(df.columns)[:15]}...")
    
    return df


def get_behavioral_stats(df, team_id, idx, window=15):
    """
    Get BEHAVIORAL stats (not outcome-based).
    Focus on HOW the team plays, not results.
    """
    prev = df.iloc[:idx]
    home_mask = prev['home_team_id'] == team_id
    away_mask = prev['visitor_team_id'] == team_id
    team_games = prev[home_mask | away_mask].tail(window)
    
    if len(team_games) < 5:
        return None
    
    # Collect behavioral metrics
    pts_for = []
    pts_against = []
    fg_pct = []
    fg3_pct = []
    reb = []
    ast = []
    wins = []
    
    # If we have detailed stats, use them
    stl = []
    blk = []
    tov = []
    pf = []
    oreb = []
    
    for _, g in team_games.iterrows():
        is_home = g['home_team_id'] == team_id
        
        if is_home:
            pts_for.append(g['pts_home'])
            pts_against.append(g['pts_away'])
            fg_pct.append(g.get('fg_pct_home', 0.45))
            fg3_pct.append(g.get('fg3_pct_home', 0.35))
            reb.append(g.get('reb_home', 42))
            ast.append(g.get('ast_home', 22))
            wins.append(1 if g['pts_home'] > g['pts_away'] else 0)
        else:
            pts_for.append(g['pts_away'])
            pts_against.append(g['pts_home'])
            fg_pct.append(g.get('fg_pct_away', 0.45))
            fg3_pct.append(g.get('fg3_pct_away', 0.35))
            reb.append(g.get('reb_away', 42))
            ast.append(g.get('ast_away', 22))
            wins.append(1 if g['pts_away'] > g['pts_home'] else 0)
        
        # Behavioral stats (estimate if not available)
        stl.append(g.get('stl', np.random.normal(7.5, 1.5)))
        blk.append(g.get('blk', np.random.normal(5.0, 1.0)))
        tov.append(g.get('to', np.random.normal(13.5, 2.0)))
        pf.append(g.get('pf', np.random.normal(20, 3)))
        oreb.append(g.get('oreb', np.random.normal(10, 2)))
    
    n = len(pts_for)
    pace = np.mean(pts_for) + np.mean(pts_against)  # Approximate pace
    
    stats = {
        # Outcome-based (for context)
        'win_pct': np.mean(wins),
        'pts_mean': np.mean(pts_for),
        'pts_against': np.mean(pts_against),
        'net_rating': np.mean(pts_for) - np.mean(pts_against),
        
        # BEHAVIORAL - Offensive
        'assist_rate': np.mean(ast) / (np.mean(fg_pct) * 80),  # Assists per made FG
        'fg_pct': np.mean(fg_pct),
        'fg3_pct': np.mean(fg3_pct),
        'oreb_rate': np.mean(oreb) / 42,  # Offensive rebounding effort
        'pace': pace,
        
        # BEHAVIORAL - Defensive  
        'steal_rate': np.mean(stl) / (np.mean(pts_against) / 100),  # Steals per 100 opp pts
        'block_rate': np.mean(blk) / (np.mean(pts_against) / 100),  # Blocks per 100 opp pts
        'forced_tov_rate': np.mean(tov) / (np.mean(pts_against) / 100),  # Approximate
        
        # BEHAVIORAL - Discipline
        'tov_rate': np.mean(tov) / (np.mean(pts_for) / 100),  # TO per 100 pts
        'foul_rate': np.mean(pf) / 48,  # Fouls per 48 min (approximate)
        
        # Trends (last 5 vs overall)
        'win_trend': np.mean(wins[-5:]) - np.mean(wins) if n >= 5 else 0,
        'pts_trend': np.mean(pts_for[-5:]) - np.mean(pts_for) if n >= 5 else 0,
        'def_trend': np.mean(pts_against) - np.mean(pts_against[-5:]) if n >= 5 else 0,  # Positive = improving
        
        'games': n,
    }
    
    return stats


def create_clash_features(df):
    """
    Create MATCHUP CLASH features.
    Focus on how behaviors interact, not absolute values.
    """
    print("\n🔧 Creating V11 Behavioral Clash Features...")
    
    features = []
    targets_ml = []
    
    for idx in range(len(df)):
        if idx < 150:
            continue
        
        row = df.iloc[idx]
        home_id = row['home_team_id']
        away_id = row['visitor_team_id']
        
        home = get_behavioral_stats(df, home_id, idx)
        away = get_behavioral_stats(df, away_id, idx)
        
        if home is None or away is None:
            continue
        
        # === CLASH FEATURES ===
        f = {}
        
        # 1. STEAL vs TURNOVER CLASH
        # High steal team vs high turnover team = exploitation potential
        f['steal_exploit_home'] = home['steal_rate'] * away['tov_rate']
        f['steal_exploit_away'] = away['steal_rate'] * home['tov_rate']
        f['steal_exploit_diff'] = f['steal_exploit_home'] - f['steal_exploit_away']
        
        # 2. BLOCK vs PAINT SCORING (use fg% as proxy)
        # High block team vs low fg% (paint-dependent) team
        f['rim_protection_home'] = home['block_rate'] * (1 - away['fg_pct'])
        f['rim_protection_away'] = away['block_rate'] * (1 - home['fg_pct'])
        f['rim_protection_diff'] = f['rim_protection_home'] - f['rim_protection_away']
        
        # 3. PACE MISMATCH
        # Who imposes their pace?
        f['pace_diff'] = home['pace'] - away['pace']
        f['pace_control'] = abs(f['pace_diff']) / 200  # Mismatch magnitude
        
        # 4. BALL MOVEMENT vs TURNOVER-FORCING
        # Good passing team vs turnover-forcing team
        f['assist_vs_steal_home'] = home['assist_rate'] / max(away['steal_rate'], 0.1)
        f['assist_vs_steal_away'] = away['assist_rate'] / max(home['steal_rate'], 0.1)
        f['playmaking_edge'] = f['assist_vs_steal_home'] - f['assist_vs_steal_away']
        
        # 5. DISCIPLINE CLASH
        # Foul-prone team vs disciplined team
        f['foul_discipline_diff'] = away['foul_rate'] - home['foul_rate']
        
        # 6. HUSTLE CLASH (offensive rebounds)
        f['hustle_diff'] = home['oreb_rate'] - away['oreb_rate']
        
        # === TREND FEATURES ===
        # Which team is improving?
        f['win_momentum_diff'] = home['win_trend'] - away['win_trend']
        f['scoring_momentum_diff'] = home['pts_trend'] - away['pts_trend']
        f['defense_momentum_diff'] = home['def_trend'] - away['def_trend']
        
        # === CONTEXT FEATURES ===
        f['win_pct_diff'] = home['win_pct'] - away['win_pct']
        f['net_rating_diff'] = home['net_rating'] - away['net_rating']
        f['sample_size'] = min(home['games'], away['games']) / 20
        
        # === RAW BEHAVIORAL (for model to learn weights) ===
        f['home_steal_rate'] = home['steal_rate']
        f['home_block_rate'] = home['block_rate']
        f['home_tov_rate'] = home['tov_rate']
        f['away_steal_rate'] = away['steal_rate']
        f['away_block_rate'] = away['block_rate']
        f['away_tov_rate'] = away['tov_rate']
        
        features.append(f)
        
        # Target
        home_pts = row['pts_home']
        away_pts = row['pts_away']
        targets_ml.append(1.0 if home_pts > away_pts else 0.0)
        
        if len(features) % 2000 == 0:
            print(f"    Processed {len(features)} games...")
    
    X = pd.DataFrame(features)
    y = np.array(targets_ml)
    
    print(f"\n  Created {len(X)} samples with {len(X.columns)} CLASH features")
    print(f"  Feature categories: Clash, Trends, Context, Raw Behavioral")
    
    return X, y


def train_v11(X, y):
    """Train V11 Behavioral Clash model."""
    print("\n🧠 Training V11 Behavioral Clash model...")
    
    # Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time-based split
    split = int(len(X) * 0.85)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # XGBoost with regularization
    print("  Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.02,
        reg_lambda=8.0, reg_alpha=2.0,  # Strong regularization
        subsample=0.8, colsample_bytree=0.6,
        min_child_weight=5,
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # LightGBM
    print("  Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.02,
        reg_lambda=8.0, reg_alpha=2.0,
        subsample=0.8, colsample_bytree=0.6,
        min_child_samples=15,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    # Ensemble
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
    ensemble_pred = 0.5 * xgb_pred + 0.5 * lgb_pred
    
    # Calibrate
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(ensemble_pred, y_test)
    
    # Evaluate
    pred_binary = (ensemble_pred > 0.5).astype(int)
    acc = accuracy_score(y_test, pred_binary)
    auc = roc_auc_score(y_test, ensemble_pred)
    
    xgb_acc = accuracy_score(y_test, (xgb_pred > 0.5).astype(int))
    lgb_acc = accuracy_score(y_test, (lgb_pred > 0.5).astype(int))
    
    print(f"\n  📊 V11 Results:")
    print(f"     XGBoost:  {xgb_acc:.1%}")
    print(f"     LightGBM: {lgb_acc:.1%}")
    print(f"     Ensemble: {acc:.1%} (AUC: {auc:.4f})")
    
    # Feature importance
    print("\n  🔑 Top 10 Most Important CLASH Features:")
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importance.head(10).iterrows():
        print(f"     {row['feature']:30} | {row['importance']:.4f}")
    
    return {
        'xgb': xgb_model,
        'lgb': lgb_model,
        'scaler': scaler,
        'calibrator': calibrator,
        'accuracy': acc,
        'auc': auc,
        'features': list(X.columns),
        'importance': importance.to_dict()
    }


def main():
    print("\n" + "="*60)
    print("🧠 V11 BEHAVIORAL CLASH MODEL")
    print("="*60)
    print("Innovation: Predict HOW behaviors interact in matchups")
    
    df = load_data()
    X, y = create_clash_features(df)
    
    result = train_v11(X, y)
    
    # Compare
    print("\n" + "="*60)
    print("📊 V11 vs V6 COMPARISON")
    print("="*60)
    v6_acc = 0.654
    v11_acc = result['accuracy']
    diff = (v11_acc - v6_acc) * 100
    symbol = '↑' if diff > 0 else '↓'
    print(f"  V6 Moneyline:  {v6_acc:.1%}")
    print(f"  V11 Moneyline: {v11_acc:.1%}")
    print(f"  Difference:    {symbol}{abs(diff):.1f}pp")
    
    # Save
    with open(MODELS_DIR / "v11_nba_clash.pkl", 'wb') as f:
        pickle.dump(result, f)
    
    with open(MODELS_DIR / "v11_nba_metrics.json", 'w') as f:
        json.dump({
            'accuracy': result['accuracy'],
            'auc': result['auc'],
            'top_features': list(importance.head(10)['feature'])
        }, f, indent=2)
    
    print(f"\n  Saved to: {MODELS_DIR / 'v11_nba_clash.pkl'}")
    
    return result


if __name__ == "__main__":
    result = main()
