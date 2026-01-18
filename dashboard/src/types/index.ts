/**
 * TypeScript interfaces for the dashboard
 */

// ESPN API Types
export interface ESPNEvent {
    id: string;
    date: string;
    status: {
        type: { name: string };
        period?: number;
        displayClock?: string;
    };
    competitions: ESPNCompetition[];
    leagueName?: string;
    leagueId?: string;
}

export interface ESPNCompetition {
    competitors: ESPNCompetitor[];
    venue?: { fullName: string };
    broadcasts?: { names: string[] }[];
    odds?: { details: string; overUnder: number }[];
}

export interface ESPNCompetitor {
    homeAway: 'home' | 'away';
    team: {
        id: string;
        abbreviation: string;
        displayName: string;
        shortDisplayName?: string;
    };
    score: string;
    records?: { summary: string }[];
    linescores?: { value: number }[];
    statistics?: { name: string; displayValue: string }[];
    leaders?: { leaders: { athlete: { shortName: string }; displayValue: string }[] }[];
}

// Processed types
export interface Team {
    name: string;
    abbreviation: string;
    score: number;
    isHome: boolean;
    record: string;
    lineScores: number[];
    statistics: { name: string; displayValue: string }[];
}

export interface GameStatus {
    isLive: boolean;
    isFinal: boolean;
    period: number;
    clock: string;
}

export interface Game {
    id: string;
    date: string;
    status: GameStatus;
    homeTeam: Team;
    awayTeam: Team;
    venue: string;
    broadcast: string;
    odds: string;
    overUnder: number | null;
    leagueName?: string;
}

// Prediction types
export interface Prediction {
    game_id: string;
    pick: string;
    confidence: number;
    pick_home: boolean;
    predictions?: {
        moneyline: { pick: string; confidence: number; pick_home: boolean };
        spread?: { pick: string; line: number; confidence: number };
        total?: { pick: string; line: number; confidence: number };
    };
}

export interface PredictionHistory {
    gameId: string;
    sport: string;
    betType: string;
    pick: string;
    confidence: number;
    odds: string;
    timestamp: string;
    result: 'pending' | 'win' | 'loss' | 'push';
    resolvedAt?: string;
}

// Config types
export interface SportTab {
    id: string;
    label: string;
    endpoint: string;
}

export interface BetType {
    id: string;
    label: string;
    title: string;
}

export interface ParlayConfig {
    legs: number;
    odds: string;
    risk: 'low' | 'medium' | 'high';
    payout: number;
}

// Component prop types
export interface NavProps {
    currentSport: string;
    onSportChange: (sport: string, endpoint: string) => void;
    liveCount: number;
}

export interface GameCardProps {
    game: Game;
    sport: string;
    prediction: Prediction | null;
    onTrack: (gameId: string, pick: string, confidence: number) => void;
}

export interface PicksProps {
    sport: string;
    pickType: string;
    onPickTypeChange: (type: string) => void;
}

export interface ParlaysProps {
    games: Game[];
    sport: string;
    predictions: Record<string, Prediction>;
}
