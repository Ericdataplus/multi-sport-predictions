
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = Path(__file__).parent.parent / "analysis_championships.md"

def load_predictions():
    with open(DATA_DIR / 'player_props_predictions.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_report():
    data = load_predictions()
    
    nfl_props = data['sports']['nfl']['predictions']
    ncaa_props = data['sports']['ncaa_football']['predictions']
    
    # Categorize
    events = {
        'National Championship': [],
        'AFC Championship': [],
        'NFC Championship': []
    }
    
    for p in nfl_props:
        grp = p.get('event_group')
        if grp in events:
            events[grp].append(p)
            
    for p in ncaa_props:
        grp = p.get('event_group')
        if grp in events:
            events[grp].append(p)
            
    # Generate MD
    lines = []
    lines.append("# Championship Games Analysis Report")
    lines.append(f"Generated at: {data.get('generated_at')}\n")
    
    for event_name, props in events.items():
        if not props:
            continue
            
        lines.append(f"## {event_name}")
        
        # Trends
        fire_props = [p for p in props if p['trend'] == '🔥']
        lines.append(f"**Total Props Evaluated**: {len(props)}")
        lines.append(f"**High Confidence Plays (🔥)**: {len(fire_props)}\n")
        
        # Top 5 Confidence
        sorted_props = sorted(props, key=lambda x: x['confidence'], reverse=True)
        lines.append("### Top 5 Key Player Trends")
        for p in sorted_props[:5]:
            line_str = f"{p['line']:.1f}" if p['line'] else "N/A"
            lines.append(f"- **{p['player']}** ({p['team']}) - {p['prop']} {p['pick']} {line_str}")
            lines.append(f"  - Confidence: {int(p['confidence']*100)}% {p.get('trend', '')}")
            if p.get('player_avg'):
                lines.append(f"  - Projected Avg: {p['player_avg']}")
            lines.append("")
            
        # Comparison (if 2 teams)
        teams = set(p['team'] for p in props)
        if len(teams) >= 2:
            lines.append("### Team Momentum")
            for t in teams:
                t_props = [p for p in props if p['team'] == t]
                avg_conf = sum(p['confidence'] for p in t_props) / len(t_props) if t_props else 0
                lines.append(f"- **{t}**: Avg Prop Confidence {int(avg_conf*100)}%")
        
        lines.append("\n---\n")
        
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
        
    print(f"Analysis saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_report()
