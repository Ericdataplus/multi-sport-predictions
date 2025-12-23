"""
Tennis Data Fetcher & V6 Model Trainer
=======================================
Uses Jeff Sackmann's free ATP/WTA data from GitHub.
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
from io import StringIO
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
DATA_DIR = BASE_DIR / "data" / "tennis"
MODELS_DIR = BASE_DIR / "models"


def fetch_tennis_data():
    """Fetch ATP match data from Jeff Sackmann's GitHub."""
    print("  Fetching tennis data from GitHub...")
    
    # ATP matches from recent years
    base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{}.csv"
    
    all_data = []
    for year in range(2018, 2025):
        url = base_url.format(year)
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                df = pd.read_csv(StringIO(resp.text))
                df['year'] = year
                all_data.append(df)
                print(f"    {year}: {len(df)} matches")
        except Exception as e:
            print(f"    {year}: Error - {e}")
    
    if not all_data:
        print("  No data fetched!")
        return None
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Save to disk
    df.to_csv(DATA_DIR / "atp_matches.csv", index=False)
    print(f"\n  Total: {len(df)} matches saved to {DATA_DIR / 'atp_matches.csv'}")
    
    return df


def load_tennis_data():
    """Load tennis data, fetching if needed."""
    path = DATA_DIR / "atp_matches.csv"
    
    if path.exists():
        print("  Loading cached tennis data...")
        return pd.read_csv(path)
    else:
        return fetch_tennis_data()


class TennisBehavioralEngine:
    """Feature engineering for tennis."""
    
    def __init__(self, df):
        self.df = df
        self.player_stats = {}
    
    def build_player_histories(self):
        """Build player match histories."""
        print("  Building player histories...")
        
        histories = {}
        
        for _, row in self.df.iterrows():
            winner_id = row['winner_id']
            loser_id = row['loser_id']
            date = row.get('tourney_date', 0)
            surface = row.get('surface', 'Hard')
            
            # Winner stats
            if winner_id not in histories:
                histories[winner_id] = []
            histories[winner_id].append({
                'date': date,
                'won': True,
                'surface': surface,
                'rank': row.get('winner_rank', 100),
                'sets_won': row.get('winner_sets', 2),
                'sets_lost': row.get('loser_sets', 0),
                'aces': row.get('w_ace', 0),
                'dfs': row.get('w_df', 0),
                'bp_saved': row.get('w_bpSaved', 0),
                'bp_faced': row.get('w_bpFaced', 1),
            })
            
            # Loser stats
            if loser_id not in histories:
                histories[loser_id] = []
            histories[loser_id].append({
                'date': date,
                'won': False,
                'surface': surface,
                'rank': row.get('loser_rank', 100),
                'sets_won': row.get('loser_sets', 0),
                'sets_lost': row.get('winner_sets', 2),
                'aces': row.get('l_ace', 0),
                'dfs': row.get('l_df', 0),
                'bp_saved': row.get('l_bpSaved', 0),
                'bp_faced': row.get('l_bpFaced', 1),
            })
        
        return histories
    
    def get_player_form(self, history, n=10, surface=None):
        """Get player form from recent matches."""
        if len(history) < 3:
            return None
        
        recent = history[-n:] if len(history) >= n else history
        
        # Surface-specific if provided
        if surface:
            surface_matches = [m for m in recent if m.get('surface') == surface]
            if len(surface_matches) >= 3:
                recent = surface_matches
        
        wins = sum(1 for m in recent if m['won'])
        n_matches = len(recent)
        
        return {
            'win_rate': wins / n_matches,
            'matches': n_matches,
            'avg_rank': np.mean([m['rank'] for m in recent if m['rank'] < 9999]),
            'sets_won_avg': np.mean([m['sets_won'] for m in recent]),
            'sets_lost_avg': np.mean([m['sets_lost'] for m in recent]),
            'aces_avg': np.mean([m.get('aces', 5) for m in recent]),
            'dfs_avg': np.mean([m.get('dfs', 2) for m in recent]),
            'bp_save_rate': sum(m.get('bp_saved', 0) for m in recent) / max(sum(m.get('bp_faced', 1) for m in recent), 1),
            'momentum': sum(1 for m in recent[-3:] if m['won']) / min(len(recent), 3),
        }
    
    def create_features(self, histories):
        """Create features for all matches."""
        print("  Creating features...")
        
        features = []
        targets = []
        
        for idx, row in self.df.iterrows():
            winner_id = row['winner_id']
            loser_id = row['loser_id']
            date = row.get('tourney_date', 0)
            surface = row.get('surface', 'Hard')
            
            # Get histories before this match
            w_hist = [m for m in histories.get(winner_id, []) if m['date'] < date]
            l_hist = [m for m in histories.get(loser_id, []) if m['date'] < date]
            
            if len(w_hist) < 3 or len(l_hist) < 3:
                continue
            
            w_form = self.get_player_form(w_hist, 15, surface)
            l_form = self.get_player_form(l_hist, 15, surface)
            
            if w_form is None or l_form is None:
                continue
            
            # Randomly assign player1/player2 (so model learns both directions)
            if np.random.random() > 0.5:
                p1_form, p2_form = w_form, l_form
                p1_won = 1
            else:
                p1_form, p2_form = l_form, w_form
                p1_won = 0
            
            f = {
                'win_rate_diff': p1_form['win_rate'] - p2_form['win_rate'],
                'rank_diff': (p2_form['avg_rank'] - p1_form['avg_rank']) / 100,  # Lower rank is better
                'sets_diff': (p1_form['sets_won_avg'] - p1_form['sets_lost_avg']) - (p2_form['sets_won_avg'] - p2_form['sets_lost_avg']),
                'aces_diff': (p1_form['aces_avg'] - p2_form['aces_avg']) / 10,
                'dfs_diff': (p2_form['dfs_avg'] - p1_form['dfs_avg']) / 5,  # More DFs is bad
                'bp_save_diff': p1_form['bp_save_rate'] - p2_form['bp_save_rate'],
                'momentum_diff': p1_form['momentum'] - p2_form['momentum'],
                'matches_diff': (p1_form['matches'] - p2_form['matches']) / 20,
                
                # Raw form
                'p1_win_rate': p1_form['win_rate'],
                'p2_win_rate': p2_form['win_rate'],
                'p1_rank': min(p1_form['avg_rank'], 500) / 100,
                'p2_rank': min(p2_form['avg_rank'], 500) / 100,
            }
            
            features.append(f)
            targets.append(p1_won)
            
            if len(features) % 5000 == 0:
                print(f"    Processed {len(features)} matches...")
        
        print(f"  Created {len(features)} samples")
        return pd.DataFrame(features), np.array(targets)


class V6TennisModel:
    """V6 Tennis model."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.calibrators = {}
        self.metrics = {}
        self.feature_names = None
    
    def train(self, X, y):
        """Train the model."""
        print("\n  Training V6 Tennis model...")
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['moneyline'] = scaler
        self.feature_names = list(X.columns)
        
        # Time-based split (80/20)
        split = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"    Train: {len(X_train)}, Test: {len(X_test)}")
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.01,
            reg_lambda=5.0, reg_alpha=1.0, subsample=0.8, colsample_bytree=0.7,
            random_state=42, n_jobs=-1, use_label_encoder=False
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.01,
            reg_lambda=5.0, reg_alpha=1.0, subsample=0.8, colsample_bytree=0.7,
            random_state=42, n_jobs=-1, verbose=-1
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        
        # Ensemble
        xgb_p = xgb_model.predict_proba(X_test)[:, 1]
        lgb_p = lgb_model.predict_proba(X_test)[:, 1]
        ensemble_p = 0.5 * xgb_p + 0.5 * lgb_p
        
        pred = (ensemble_p >= 0.5).astype(int)
        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, ensemble_p)
        
        print(f"    Accuracy: {acc:.1%}")
        print(f"    AUC: {auc:.4f}")
        
        self.models['moneyline'] = {'xgb': xgb_model, 'lgb': lgb_model}
        self.calibrators['moneyline'] = IsotonicRegression(out_of_bounds='clip').fit(ensemble_p, y_test)
        self.metrics['moneyline'] = {'accuracy': acc, 'auc': auc, 'test_size': len(y_test)}
        
        # Copy for contracts (same model)
        self.models['contracts'] = self.models['moneyline']
        self.calibrators['contracts'] = self.calibrators['moneyline']
        self.metrics['contracts'] = self.metrics['moneyline']
        
        return acc
    
    def save(self):
        """Save model."""
        path = MODELS_DIR / "v6_tennis_complete.pkl"
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'calibrators': self.calibrators,
                'metrics': self.metrics,
                'feature_names': self.feature_names,
            }, f)
        
        metrics_path = MODELS_DIR / "v6_tennis_metrics.json"
        json_metrics = {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv 
                           for kk, vv in v.items()} 
                       for k, v in self.metrics.items()}
        with open(metrics_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"\n  Saved to: {path}")


def train_v6_tennis():
    """Train V6 Tennis model."""
    print("\n" + "="*60)
    print("V6 TENNIS BEHAVIORAL PROXY MODEL")
    print("="*60)
    
    df = load_tennis_data()
    if df is None or len(df) == 0:
        return None
    
    print(f"  Loaded {len(df)} matches")
    
    engine = TennisBehavioralEngine(df)
    histories = engine.build_player_histories()
    X, y = engine.create_features(histories)
    
    if len(X) == 0:
        print("  No valid features created!")
        return None
    
    print(f"\n  Features: {X.shape}")
    
    model = V6TennisModel()
    acc = model.train(X, y)
    
    print("\n" + "="*60)
    print("V6 TENNIS RESULTS")
    print("="*60)
    for bt, m in model.metrics.items():
        print(f"  {bt.upper():12} | Accuracy: {m['accuracy']:.1%} | AUC: {m['auc']:.4f}")
    
    # Baseline
    base_rate = np.mean(y)
    print(f"\n  Baseline (random): 50.0%")
    print(f"  Favorite wins: ~{max(base_rate, 1-base_rate)*100:.1f}%")
    
    model.save()
    return model


if __name__ == "__main__":
    model = train_v6_tennis()
