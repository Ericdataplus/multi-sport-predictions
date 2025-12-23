"""
V7 Neural Network Model for NBA
================================
Inspired by kyleskom/NBA-Machine-Learning-Sports-Betting (69% accuracy)

Key differences from V6:
1. Neural Network (TensorFlow) instead of XGBoost ensemble
2. Uses cumulative season-to-date stats (like kyleskom)
3. More raw features (40+ stats)
4. Deeper network with dropout for regularization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import tensorflow as tf

# TF 2.16+ uses tf.keras
keras = tf.keras
layers = tf.keras.layers

# Also try XGBoost with their exact params for comparison
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "nba"
MODELS_DIR = BASE_DIR / "models"


class NBASeasonStatsEngine:
    """
    Feature engineering using cumulative season stats approach.
    This matches what kyleskom uses - season-to-date averages.
    """
    
    def load_data(self):
        """Load NBA game data."""
        print("  Loading NBA data...")
        
        # Use games.csv which has the full stats
        path = DATA_DIR / "games.csv"
        df = pd.read_csv(path)
        
        # Standardize column names (lowercase)
        df.columns = [c.lower() for c in df.columns]
        
        # Parse dates
        df['date'] = pd.to_datetime(df['game_date_est'])
        
        # Filter to completed games with scores
        df = df.dropna(subset=['pts_home', 'pts_away'])
        df = df[df['date'].dt.year >= 2012]
        
        # Add season column
        df['season'] = df['date'].apply(lambda x: x.year if x.month >= 10 else x.year - 1)
        
        print(f"  Loaded {len(df)} games")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Columns: {list(df.columns)[:15]}...")
        
        return df.sort_values('date').reset_index(drop=True)
    
    def calculate_season_stats(self, team_id, games_before_date, season):
        """
        Calculate cumulative season-to-date stats for a team.
        This is the key insight from kyleskom - use rolling season averages.
        """
        season_games = games_before_date[games_before_date['season'] == season]
        
        if len(season_games) < 3:
            return None
        
        # Aggregate stats
        stats = {
            'games_played': len(season_games),
            'wins': 0,
            'pts_mean': 0,
            'pts_against_mean': 0,
            'fg_pct_mean': 0,
            'fg3_pct_mean': 0,
            'ft_pct_mean': 0,
            'reb_mean': 0,
            'ast_mean': 0,
            'stl_mean': 0,
            'blk_mean': 0,
            'tov_mean': 0,
            'pf_mean': 0,
        }
        
        pts_list = []
        pts_against_list = []
        fg_pct_list = []
        fg3_pct_list = []
        ft_pct_list = []
        reb_list = []
        ast_list = []
        stl_list = []
        blk_list = []
        tov_list = []
        pf_list = []
        
        for _, game in season_games.iterrows():
            # Determine if team was home or away
            if game.get('home_team_id') == team_id:
                is_home = True
                suffix = '_home'
                opp_suffix = '_away'
            elif game.get('visitor_team_id') == team_id or game.get('away_team_id') == team_id:
                is_home = False
                suffix = '_away'
                opp_suffix = '_home'
            else:
                continue
            
            pts = game.get(f'pts{suffix}', game.get('pts_home' if is_home else 'pts_away', 0))
            pts_against = game.get(f'pts{opp_suffix}', game.get('pts_away' if is_home else 'pts_home', 0))
            
            if pd.notna(pts) and pd.notna(pts_against):
                pts_list.append(pts)
                pts_against_list.append(pts_against)
                if pts > pts_against:
                    stats['wins'] += 1
            
            # Get other stats
            fg_pct = game.get(f'fg_pct{suffix}', game.get('fg_pct_home' if is_home else 'fg_pct_away', 0))
            if pd.notna(fg_pct) and fg_pct > 0:
                fg_pct_list.append(fg_pct)
            
            fg3_pct = game.get(f'fg3_pct{suffix}', game.get('fg3_pct_home' if is_home else 'fg3_pct_away', 0))
            if pd.notna(fg3_pct) and fg3_pct > 0:
                fg3_pct_list.append(fg3_pct)
            
            ft_pct = game.get(f'ft_pct{suffix}', game.get('ft_pct_home' if is_home else 'ft_pct_away', 0))
            if pd.notna(ft_pct) and ft_pct > 0:
                ft_pct_list.append(ft_pct)
            
            reb = game.get(f'reb{suffix}', game.get('reb_home' if is_home else 'reb_away', 40))
            if pd.notna(reb):
                reb_list.append(reb)
            
            ast = game.get(f'ast{suffix}', game.get('ast_home' if is_home else 'ast_away', 22))
            if pd.notna(ast):
                ast_list.append(ast)
            
            stl = game.get(f'stl{suffix}', game.get('stl_home' if is_home else 'stl_away', 7))
            if pd.notna(stl):
                stl_list.append(stl)
            
            blk = game.get(f'blk{suffix}', game.get('blk_home' if is_home else 'blk_away', 5))
            if pd.notna(blk):
                blk_list.append(blk)
            
            tov = game.get(f'tov{suffix}', game.get('tov_home' if is_home else 'tov_away', 14))
            if pd.notna(tov):
                tov_list.append(tov)
            
            pf = game.get(f'pf{suffix}', game.get('pf_home' if is_home else 'pf_away', 20))
            if pd.notna(pf):
                pf_list.append(pf)
        
        if not pts_list:
            return None
        
        stats['pts_mean'] = np.mean(pts_list)
        stats['pts_against_mean'] = np.mean(pts_against_list)
        stats['win_pct'] = stats['wins'] / stats['games_played']
        stats['fg_pct_mean'] = np.mean(fg_pct_list) if fg_pct_list else 0.45
        stats['fg3_pct_mean'] = np.mean(fg3_pct_list) if fg3_pct_list else 0.35
        stats['ft_pct_mean'] = np.mean(ft_pct_list) if ft_pct_list else 0.75
        stats['reb_mean'] = np.mean(reb_list) if reb_list else 42
        stats['ast_mean'] = np.mean(ast_list) if ast_list else 22
        stats['stl_mean'] = np.mean(stl_list) if stl_list else 7
        stats['blk_mean'] = np.mean(blk_list) if blk_list else 5
        stats['tov_mean'] = np.mean(tov_list) if tov_list else 14
        stats['pf_mean'] = np.mean(pf_list) if pf_list else 20
        
        # Derived stats
        stats['net_rating'] = stats['pts_mean'] - stats['pts_against_mean']
        stats['ast_to_tov'] = stats['ast_mean'] / max(stats['tov_mean'], 1)
        
        return stats
    
    def create_features(self, df):
        """Create features for all games."""
        print("  Creating season-to-date features...")
        
        features = []
        targets = {'moneyline': [], 'spread': [], 'total': []}
        
        for idx, row in df.iterrows():
            date = row['date']
            season = row['season']
            
            # Get team IDs
            home_id = row.get('home_team_id', row.get('id_home', None))
            away_id = row.get('visitor_team_id', row.get('away_team_id', row.get('id_away', None)))
            
            if pd.isna(home_id) or pd.isna(away_id):
                continue
            
            # Games before this date
            games_before = df[df['date'] < date]
            
            # Get season stats
            home_stats = self.calculate_season_stats(home_id, games_before, season)
            away_stats = self.calculate_season_stats(away_id, games_before, season)
            
            if home_stats is None or away_stats is None:
                continue
            
            # Create feature vector (like kyleskom - use raw stats for both teams)
            f = {
                # Home team stats
                'home_win_pct': home_stats['win_pct'],
                'home_pts': home_stats['pts_mean'],
                'home_pts_against': home_stats['pts_against_mean'],
                'home_fg_pct': home_stats['fg_pct_mean'],
                'home_fg3_pct': home_stats['fg3_pct_mean'],
                'home_ft_pct': home_stats['ft_pct_mean'],
                'home_reb': home_stats['reb_mean'],
                'home_ast': home_stats['ast_mean'],
                'home_stl': home_stats['stl_mean'],
                'home_blk': home_stats['blk_mean'],
                'home_tov': home_stats['tov_mean'],
                'home_pf': home_stats['pf_mean'],
                'home_net_rating': home_stats['net_rating'],
                'home_ast_to_tov': home_stats['ast_to_tov'],
                'home_games': home_stats['games_played'],
                
                # Away team stats
                'away_win_pct': away_stats['win_pct'],
                'away_pts': away_stats['pts_mean'],
                'away_pts_against': away_stats['pts_against_mean'],
                'away_fg_pct': away_stats['fg_pct_mean'],
                'away_fg3_pct': away_stats['fg3_pct_mean'],
                'away_ft_pct': away_stats['ft_pct_mean'],
                'away_reb': away_stats['reb_mean'],
                'away_ast': away_stats['ast_mean'],
                'away_stl': away_stats['stl_mean'],
                'away_blk': away_stats['blk_mean'],
                'away_tov': away_stats['tov_mean'],
                'away_pf': away_stats['pf_mean'],
                'away_net_rating': away_stats['net_rating'],
                'away_ast_to_tov': away_stats['ast_to_tov'],
                'away_games': away_stats['games_played'],
            }
            
            features.append(f)
            
            # Targets
            home_pts = row['pts_home']
            away_pts = row['pts_away']
            
            targets['moneyline'].append(1 if home_pts > away_pts else 0)
            
            margin = home_pts - away_pts
            pred_spread = -(home_stats['net_rating'] - away_stats['net_rating']) * 0.3
            targets['spread'].append(1 if margin > pred_spread else 0)
            
            total = home_pts + away_pts
            pred_total = (home_stats['pts_mean'] + away_stats['pts_mean']) * 0.95
            targets['total'].append(1 if total > pred_total else 0)
            
            if len(features) % 2000 == 0:
                print(f"    Processed {len(features)} games...")
        
        print(f"  Created {len(features)} feature rows")
        return pd.DataFrame(features), targets


class V7NeuralNetModel:
    """Neural Network model inspired by kyleskom."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.feature_names = None
    
    def build_neural_network(self, input_dim):
        """Build deep neural network."""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(16, activation='relu'),
            layers.Dense(2, activation='softmax')  # Binary classification
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_bet_style(self, X, y, bet_type='moneyline'):
        """Train neural network for a bet type."""
        print(f"\n  Training {bet_type.upper()} with Neural Network...")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[bet_type] = scaler
        self.feature_names = list(X.columns)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, np.array(y), test_size=0.15, random_state=42
        )
        
        print(f"    Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Build and train neural network
        nn_model = self.build_neural_network(X_scaled.shape[1])
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=20, restore_best_weights=True
        )
        
        history = nn_model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=64,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        nn_pred_proba = nn_model.predict(X_test, verbose=0)
        nn_pred = np.argmax(nn_pred_proba, axis=1)
        nn_acc = accuracy_score(y_test, nn_pred)
        nn_auc = roc_auc_score(y_test, nn_pred_proba[:, 1])
        
        print(f"    Neural Network: {nn_acc:.1%} accuracy, AUC: {nn_auc:.4f}")
        
        # Also train XGBoost with kyleskom's params for comparison
        xgb_model = xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.01,
            n_estimators=750,
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        
        print(f"    XGBoost (kyleskom params): {xgb_acc:.1%} accuracy")
        
        # Use the better model
        if nn_acc >= xgb_acc:
            self.models[bet_type] = {'type': 'neural_network', 'model': nn_model}
            best_acc = nn_acc
            best_auc = nn_auc
            print(f"    → Using Neural Network")
        else:
            self.models[bet_type] = {'type': 'xgboost', 'model': xgb_model}
            best_acc = xgb_acc
            best_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
            print(f"    → Using XGBoost")
        
        self.metrics[bet_type] = {'accuracy': best_acc, 'auc': best_auc, 'test_size': len(y_test)}
        
        return best_acc
    
    def train_all(self, X, targets):
        """Train all bet types."""
        self.train_bet_style(X, targets['moneyline'], 'moneyline')
        self.train_bet_style(X, targets['spread'], 'spread')
        self.train_bet_style(X, targets['total'], 'total')
        self.train_bet_style(X, targets['moneyline'], 'contracts')
    
    def save(self):
        """Save model."""
        path = MODELS_DIR / "v7_nba_neural.pkl"
        
        # For neural networks, save weights separately
        for bet_type, model_info in self.models.items():
            if model_info['type'] == 'neural_network':
                model_info['model'].save(MODELS_DIR / f"v7_nba_{bet_type}_nn.keras")
                model_info['model'] = None  # Can't pickle Keras model directly
        
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'metrics': self.metrics,
                'feature_names': self.feature_names,
            }, f)
        
        # Save metrics JSON
        metrics_path = MODELS_DIR / "v7_nba_metrics.json"
        json_metrics = {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv 
                           for kk, vv in v.items()} 
                       for k, v in self.metrics.items()}
        with open(metrics_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"\n  Saved to: {path}")


def train_v7_nba():
    """Train V7 NBA model."""
    print("\n" + "="*60)
    print("V7 NBA NEURAL NETWORK MODEL")
    print("="*60)
    print("Inspired by kyleskom/NBA-ML-Betting (69% claimed)")
    
    engine = NBASeasonStatsEngine()
    
    df = engine.load_data()
    if df is None or len(df) == 0:
        return None
    
    X, targets = engine.create_features(df)
    
    print(f"\n  Features: {X.shape}")
    print(f"  Columns: {list(X.columns)}")
    
    model = V7NeuralNetModel()
    model.train_all(X, targets)
    
    print("\n" + "="*60)
    print("V7 NBA RESULTS")
    print("="*60)
    for bt, m in model.metrics.items():
        print(f"  {bt.upper():12} | Accuracy: {m['accuracy']:.1%} | AUC: {m['auc']:.4f}")
    
    # Compare to V6
    print("\n  Comparison to V6:")
    v6_results = {'moneyline': 0.654, 'spread': 0.734, 'total': 0.55}
    for bt in ['moneyline', 'spread', 'total']:
        v6 = v6_results.get(bt, 0)
        v7 = model.metrics.get(bt, {}).get('accuracy', 0)
        diff = (v7 - v6) * 100
        symbol = '↑' if diff > 0 else '↓' if diff < 0 else '='
        print(f"    {bt}: V6={v6:.1%} → V7={v7:.1%} ({symbol}{abs(diff):.1f}pp)")
    
    model.save()
    return model


if __name__ == "__main__":
    # Set TensorFlow to be less verbose
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    model = train_v7_nba()
