/**
 * Configuration and Constants
 */

import type { SportTab, BetType, ParlayConfig } from '../types';

export const ESPN_BASE = 'https://site.api.espn.com/apis/site/v2/sports';

// Model accuracy data per sport + bet type
export const MODEL_DATA: Record<string, Record<string, number>> = {
    nba: { moneyline: 0.65, spread: 0.73, overunder: 0.62, contracts: 0.65 },
    nfl: { moneyline: 0.65, spread: 0.69, overunder: 0.56, contracts: 0.65 },
    nhl: { moneyline: 0.72, spread: 0.67, overunder: 0.60, contracts: 0.72 },
    mlb: { moneyline: 0.58, spread: 0.62, overunder: 0.58, contracts: 0.58 },
    ncaa_basketball: { moneyline: 0.65, contracts: 0.65 },
    soccer: { moneyline: 0.67, spread: 0.75, overunder: 0.62, contracts: 0.67 },
    tennis: { moneyline: 0.63, contracts: 0.63 }
};

// Sport display names
export const SPORT_NAMES: Record<string, string> = {
    nba: '🏀 NBA',
    ncaa_basketball: '🏀 NCAA Basketball',
    nfl: '🏈 NFL',
    ncaa_football: '🏈 College Football',
    nhl: '🏒 NHL',
    mlb: '⚾ MLB',
    tennis: '🎾 Tennis',
    soccer: '⚽ Soccer'
};

// Sport tabs configuration
export const SPORT_TABS: SportTab[] = [
    { id: 'nba', label: '🏀 NBA', endpoint: '/basketball/nba/scoreboard' },
    { id: 'ncaa_basketball', label: '🏀 NCAA', endpoint: '/basketball/mens-college-basketball/scoreboard' },
    { id: 'nfl', label: '🏈 NFL', endpoint: '/football/nfl/scoreboard' },
    { id: 'ncaa_football', label: '🏈 CFB', endpoint: '/football/college-football/scoreboard' },
    { id: 'nhl', label: '🏒 NHL', endpoint: '/hockey/nhl/scoreboard' },
    { id: 'mlb', label: '⚾ MLB', endpoint: '/baseball/mlb/scoreboard' },
    { id: 'tennis', label: '🎾 Tennis', endpoint: '/tennis/atp/scoreboard' },
    { id: 'soccer', label: '⚽ Soccer', endpoint: '/soccer/eng.1/scoreboard' }
];

// Soccer leagues
export const SOCCER_LEAGUES = [
    { id: 'eng.1', name: '🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League' },
    { id: 'esp.1', name: '🇪🇸 La Liga' },
    { id: 'ger.1', name: '🇩🇪 Bundesliga' },
    { id: 'ita.1', name: '🇮🇹 Serie A' },
    { id: 'fra.1', name: '🇫🇷 Ligue 1' },
    { id: 'uefa.champions', name: '🏆 Champions League' }
];

// Bet type tabs
export const BET_TYPES: BetType[] = [
    { id: 'moneyline', label: 'Moneyline', title: 'Pick the winner' },
    { id: 'spread', label: 'Spread', title: 'Win by enough points' },
    { id: 'total', label: 'O/U', title: 'Over or Under total' },
    { id: 'props', label: 'Player Props', title: 'Individual player stats' },
    { id: 'contracts', label: '📈 Contracts', title: 'Trade like stocks' },
    { id: 'history', label: '📜 History', title: 'Past predictions' }
];

// Bet explainers
export const BET_EXPLAINERS: Record<string, string> = {
    moneyline: '💡 <strong>Moneyline</strong> = Pick who wins. Simplest bet!',
    spread: '💡 <strong>Spread</strong> = Win by enough points.',
    total: '💡 <strong>Over/Under</strong> = Total score over or under a line.',
    props: '💡 <strong>Player Props</strong> = Bet on individual player stats (e.g., Yards, TDs).',
    contracts: '💡 <strong>Contracts</strong> = Trade predictions like stocks!',
    history: '📜 Your tracked predictions and results.'
};

// Parlay configs
export const PARLAY_CONFIGS: ParlayConfig[] = [
    { legs: 2, odds: '+264', risk: 'low', payout: 364 },
    { legs: 3, odds: '+595', risk: 'medium', payout: 695 },
    { legs: 4, odds: '+1228', risk: 'high', payout: 1328 }
];

// Refresh interval
export const REFRESH_INTERVAL = 30000;
