"""
Download NBA Player Props Data
==============================
Dedicated script solely for robust data downloading.
Tries multiple endpoints to get player game logs validation.
"""
import requests
import pandas as pd
import time
from pathlib import Path

def download_nba_data(season="2024-25"):
    print(f"🏀 Fetching NBA Player Game Logs for {season}...")
    
    # Headers are critical for NBA API
    headers = {
        'Host': 'stats.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
        'Accept': 'application/json',
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com',
        'x-nba-stats-token': 'true',
        'x-nba-stats-origin': 'stats',
    }

    # Endpoint: LeagueGameLog (best for bulk)
    url = "https://stats.nba.com/stats/leaguegamelog"
    params = {
        'Counter': '1000',
        'DateFrom': '',
        'DateTo': '',
        'Direction': 'DESC',
        'LeagueID': '00',
        'PlayerOrTeam': 'P', # P for Player
        'Season': season,
        'SeasonType': 'Regular Season',
        'Sorter': 'DATE',
    }

    try:
        # Retry loop
        for i in range(3):
            try:
                print(f"   Attempt {i+1}/3...")
                resp = requests.get(url, headers=headers, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                print(f"   ⚠️ Error: {e}")
                time.sleep(5)
        else:
            print("❌ Failed after 3 attempts.")
            return

        headers_list = data['resultSets'][0]['headers']
        rows = data['resultSets'][0]['rowSet']
        df = pd.DataFrame(rows, columns=headers_list)
        
        save_path = Path("data/nba/player_props_2024.csv")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✅ Success! Saved {len(df)} records to {save_path}")
        print("   Columns:", list(df.columns[:5]), "...")
        
    except Exception as e:
        print(f"❌ Fatal Error: {e}")

if __name__ == "__main__":
    download_nba_data()
