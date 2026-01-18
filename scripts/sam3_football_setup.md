# SAM 3 Football Player Behavior Analysis

## Goal
Use SAM 3 to track player behavior from video to extract features that traditional stats don't capture.

## Hardware
- GPU: RTX 3060 12GB VRAM
- Likely enough for inference, marginal for fine-tuning

## Phase 1: Setup & Test (1-2 hours)

### 1. Install SAM 3
```bash
conda create -n sam3 python=3.12
conda activate sam3

# Install PyTorch with CUDA
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Clone SAM 3
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e ".[notebooks]"

# Authenticate with HuggingFace (need to request access first)
pip install huggingface_hub
huggingface-cli login
```

### 2. Request Model Access
Go to: https://huggingface.co/facebook/sam3
Click "Request Access" - usually approved within hours

### 3. Test with sample video
```python
from sam3 import SAM3VideoPredictor

predictor = SAM3VideoPredictor.from_pretrained("facebook/sam3")

# Test with a short clip
predictor.set_video("sample_football_clip.mp4")

# Segment all players
results = predictor.predict(text="football players")

# Track specific concepts
home_team = predictor.predict(text="players in white jerseys")
away_team = predictor.predict(text="players in red jerseys")
ball = predictor.predict(text="football")
```

## Phase 2: Feature Extraction Pipeline

### Features to Extract:
1. **Player positions over time** (x, y coordinates per frame)
2. **Player speeds** (distance between frames / time)
3. **Formation patterns** (clustering of player positions)
4. **Ball possession** (which player segment overlaps with ball)
5. **Distance covered** (cumulative movement per player)
6. **Spacing metrics** (average distance to nearest teammate/opponent)

### Example Feature Extraction:
```python
import numpy as np
from collections import defaultdict

def extract_player_features(video_path, predictor):
    """Extract behavioral features from football video."""
    
    predictor.set_video(video_path)
    
    # Track all players
    player_tracks = predictor.predict(text="football player", track=True)
    ball_track = predictor.predict(text="football", track=True)
    
    features = defaultdict(list)
    
    for frame_idx, frame_data in enumerate(player_tracks):
        # Get centroids of each player mask
        for player_id, mask in frame_data['masks'].items():
            centroid = get_centroid(mask)
            features[player_id].append({
                'frame': frame_idx,
                'x': centroid[0],
                'y': centroid[1],
                'area': mask.sum()  # Player size in pixels
            })
    
    # Calculate derived features
    for player_id, positions in features.items():
        speeds = []
        for i in range(1, len(positions)):
            dx = positions[i]['x'] - positions[i-1]['x']
            dy = positions[i]['y'] - positions[i-1]['y']
            speed = np.sqrt(dx**2 + dy**2)
            speeds.append(speed)
        
        features[player_id] = {
            'avg_speed': np.mean(speeds),
            'max_speed': np.max(speeds),
            'total_distance': sum(speeds),
            'position_variance': np.var([p['x'] for p in positions])
        }
    
    return features

def get_centroid(mask):
    """Get center of mass of a segmentation mask."""
    y, x = np.where(mask)
    return (x.mean(), y.mean())
```

## Phase 3: Connect to Prediction Model

### Aggregate game-level features:
```python
def game_level_features(all_player_features):
    """Aggregate player-level to game-level features."""
    
    home_players = [f for pid, f in all_player_features.items() if is_home(pid)]
    away_players = [f for pid, f in all_player_features.items() if is_away(pid)]
    
    return {
        'home_avg_speed': np.mean([p['avg_speed'] for p in home_players]),
        'away_avg_speed': np.mean([p['avg_speed'] for p in away_players]),
        'home_total_distance': sum([p['total_distance'] for p in home_players]),
        'away_total_distance': sum([p['total_distance'] for p in away_players]),
        'speed_differential': home_avg - away_avg,
        # ... more features
    }
```

### Feed to XGBoost:
```python
# Combine video features with traditional stats
combined_features = {
    **traditional_stats,  # Points, yards, turnovers
    **video_features      # Speed, distance, formations
}

# Train model
model = XGBClassifier()
model.fit(X_combined, y)
```

## Phase 4: Data Sources

### Where to get football video:
1. **NFL Game Pass** - Full game footage (paid)
2. **YouTube highlights** - Free but edited
3. **All-22 film** - Best for tactical analysis
4. **College football** - More accessible

### Recommended: Start with YouTube highlights
- 3-5 minute clips per game
- Enough to test the pipeline
- Free and accessible

## VRAM Optimization Tips

If you hit OOM errors:
```python
# Use smaller model variant if available
predictor = SAM3VideoPredictor.from_pretrained("facebook/sam3-tiny")

# Process in smaller chunks
predictor.set_video(video_path, max_frames=100)  

# Lower resolution
predictor.set_video(video_path, resolution=480)

# Use gradient checkpointing for fine-tuning
predictor.enable_gradient_checkpointing()

# Mixed precision
predictor.half()  # FP16 inference
```

## Expected Outcome

If this works, you could have features like:
- Pre-game energy levels (warmup footage)
- In-game hustle metrics
- Fatigue indicators (speed over time)
- Formation tendencies

These would be **unique alpha** that no one else is using for predictions!
