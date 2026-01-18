/**
 * Utility helper functions
 */

import type { ESPNEvent, ESPNCompetitor, Game, Team, GameStatus } from '../types';

// Seeded random (deterministic)
export function seededRandom(seed: string): number {
    let hash = 0;
    for (let i = 0; i < seed.length; i++) {
        const char = seed.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash;
    }
    const x = Math.sin(hash) * 10000;
    return x - Math.floor(x);
}

// Format game time
export function formatGameTime(dateStr: string): string {
    const gameTime = new Date(dateStr);
    const now = new Date();
    const isToday = gameTime.toDateString() === now.toDateString();
    const isTomorrow = gameTime.toDateString() === new Date(now.getTime() + 86400000).toDateString();

    const timeStr = gameTime.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });

    if (isToday) return `Today ${timeStr}`;
    if (isTomorrow) return `Tomorrow ${timeStr}`;

    return gameTime.toLocaleDateString('en-US', {
        weekday: 'short', month: 'short', day: 'numeric'
    }) + ` ${timeStr}`;
}

// Get game status
export function getGameStatus(event: ESPNEvent): GameStatus {
    const status = event.status?.type?.name || '';
    return {
        isLive: status.includes('IN_PROGRESS'),
        isFinal: status.includes('FINAL'),
        period: event.status?.period || 0,
        clock: event.status?.displayClock || ''
    };
}

// Extract team info
export function extractTeam(competitor: ESPNCompetitor): Team {
    return {
        name: competitor?.team?.shortDisplayName || competitor?.team?.displayName || 'Unknown',
        abbreviation: competitor?.team?.abbreviation || '???',
        score: parseInt(competitor?.score) || 0,
        isHome: competitor?.homeAway === 'home',
        record: competitor?.records?.[0]?.summary || '',
        lineScores: (competitor?.linescores || []).map(ls => ls.value),
        statistics: competitor?.statistics || []
    };
}

// Transform ESPN event to Game
export function transformEvent(event: ESPNEvent): Game | null {
    const comp = event.competitions?.[0];
    if (!comp || comp.competitors.length < 2) return null;

    const homeComp = comp.competitors.find(c => c.homeAway === 'home') || comp.competitors[0];
    const awayComp = comp.competitors.find(c => c.homeAway === 'away') || comp.competitors[1];

    return {
        id: event.id,
        date: event.date,
        status: getGameStatus(event),
        homeTeam: extractTeam(homeComp),
        awayTeam: extractTeam(awayComp),
        venue: comp.venue?.fullName || '',
        broadcast: comp.broadcasts?.[0]?.names?.[0] || '',
        odds: comp.odds?.[0]?.details || '',
        overUnder: comp.odds?.[0]?.overUnder || null,
        leagueName: event.leagueName
    };
}

// Get stat value
export function getStatValue(stats: { name: string; displayValue: string }[], name: string): string {
    return stats.find(s => s.name === name)?.displayValue || '-';
}
