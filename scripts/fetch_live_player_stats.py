"""
Live Player Stats Fetcher (NFL & NCAA)
======================================
Fetches real-time player statistics from ESPN API for NFL and NCAA Football player props.
Updates the player_props_predictions.json with live data.
"""

import json
import requests
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# ESPN API endpoints
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football"

# Championship teams configuration
NFL_TEAMS = {
    'DEN': 'Denver Broncos',
    'HOU': 'Houston Texans', 
    'LAR': 'Los Angeles Rams',
    'SF': 'San Francisco 49ers'
}

NCAA_TEAMS = {
    'MIA': 'Miami Hurricanes',
    'IND': 'Indiana Hoosiers'
}

# Hardcoded IDs in case lookup fails (often needed for college)
TEAM_IDS = {
    'MIA': '2390',
    'IND': '84',
    'DEN': '7',
    'HOU': '34',
    'LAR': '14',
    'SF': '25'
}

def get_active_players(team_abbrev, sport='nfl'):
    """Fetch active players (Starters) from Depth Chart or Roster."""
    league = 'college-football' if sport == 'ncaa' else 'nfl'
    base_url = f"{ESPN_BASE}/{league}/teams"
    
    try:
        team_id = TEAM_IDS.get(team_abbrev)
        
        # dynamic lookup if not hardcoded (optional, but good for robustness)
        if not team_id:
            try:
                # This might be heavy for NCAA, skipping dynamic lookup for now to rely on hardcoded
                pass 
            except:
                pass

        if not team_id:
            print(f"   [!] Team ID for {team_abbrev} not found")
            return []
            
        print(f"   Using Team ID: {team_id} ({sport.upper()})")

        players = []
        seen_ids = set()

        # 1. Try Depth Chart (Starters) - Preferred
        dc_url = f"{base_url}/{team_id}/depthcharts"
        try:
            resp = requests.get(dc_url, timeout=10)
            if resp.status_code == 200:
                dc_data = resp.json()
                for item in dc_data.get('items', []):
                    for position_group in item.get('positions', {}).values():
                        if isinstance(position_group, list):
                             for entry in position_group:
                                 athlete = entry.get('athlete', {})
                                 rank = entry.get('rank')
                                 slot = entry.get('slot')
                                 
                                 # Starters (rank 1) or Slot 1/2
                                 if rank == 1 or (slot in [1, 2] if slot else False):
                                     aid = athlete.get('id')
                                     if aid and aid not in seen_ids:
                                         seen_ids.add(aid)
                                         name = athlete.get('displayName') or athlete.get('fullName')
                                         pos = entry.get('position', {}).get('abbreviation') or 'UNK'
                                         
                                         if pos in ['QB', 'RB', 'WR', 'TE']:
                                             players.append({
                                                 'id': aid,
                                                 'name': name,
                                                 'position': pos,
                                                 'team': team_abbrev,
                                                 'sport': sport,
                                                 'status': 'Starter'
                                             })
        except Exception:
            pass # DC failed, continue to roster

        if players:
            return players

        # 2. Fallback to Full Roster
        print(f"   [!] Fallback to roster for {team_abbrev}")
        roster_url = f"{base_url}/{team_id}/roster"
        resp = requests.get(roster_url, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            for athlete_group in data.get('athletes', []):
                 for player in athlete_group.get('items', []):
                     # For NCAA, active status might be managed differently, but usually 'type' works
                     # Also allow empty status for NCAA if name exists
                     status_type = player.get('status', {}).get('type')
                     if status_type == 'active' or sport == 'ncaa': # Relax for NCAA
                         pos = player.get('position', {}).get('abbreviation')
                         if pos in ['QB', 'RB', 'WR', 'TE']:
                             players.append({
                                 'id': player.get('id'),
                                 'name': player.get('displayName'),
                                 'position': pos,
                                 'team': team_abbrev,
                                 'sport': sport,
                                 'status': 'Active'
                             })
        
        return players

    except Exception as e:
        print(f"   [!] Error fetching players for {team_abbrev}: {e}")
        return []

def fetch_live_stats():
    print("=" * 60)
    print("LIVE PLAYER STATS FETCHER (NFL & NCAA)")
    print("=" * 60)
    
    all_players = {}
    
    # NFL
    for abbrev, name in NFL_TEAMS.items():
        print(f"\n[{abbrev}] Fetching {name} active players...")
        players = get_active_players(abbrev, sport='nfl')
        
        # Sort/Filter
        starters = [p for p in players if p.get('status') == 'Starter']
        others = [p for p in players if p.get('status') != 'Starter']
        final_list = (starters + others)[:40] # Cap 40
        
        print(f"   Found {len(final_list)} relevant players")
        for p in final_list:
             all_players[p['name']] = p

    # NCAA
    for abbrev, name in NCAA_TEAMS.items():
        print(f"\n[{abbrev}] Fetching {name} active players...")
        players = get_active_players(abbrev, sport='ncaa')
        
        starters = [p for p in players if p.get('status') == 'Starter']
        others = [p for p in players if p.get('status') != 'Starter']
        final_list = (starters + others)[:40]
        
        print(f"   Found {len(final_list)} relevant players")
        for p in final_list:
             all_players[p['name']] = p

    return all_players


def update_training_data(live_players):
    """Update training data with live player list."""
    # Combine team lists for metadata
    all_teams = {**NFL_TEAMS, **NCAA_TEAMS}
    
    output = {
        'fetched_at': datetime.now().isoformat(),
        'teams': all_teams,
        'players': live_players
    }
    
    output_path = DATA_DIR / 'live_player_data.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SAVE] Saved {len(live_players)} players to {output_path}")
    return output


if __name__ == "__main__":
    players = fetch_live_stats()
    update_training_data(players)
    print("\n" + "=" * 60)
    print("FETCH COMPLETE!")
    print("=" * 60)
