import { useState, useEffect, useCallback } from 'react';
import { ESPN_BASE, SOCCER_LEAGUES } from '../config/constants';
import { transformEvent } from '../utils/helpers';
import type { ESPNEvent, Game } from '../types';

/**
 * Hook to fetch games from ESPN API
 */
export function useGames(sport: string, endpoint: string) {
    const [games, setGames] = useState<Game[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchGames = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            let events: ESPNEvent[] = [];

            if (sport === 'soccer') {
                const results = await Promise.allSettled(
                    SOCCER_LEAGUES.map(async (league) => {
                        const res = await fetch(`${ESPN_BASE}/soccer/${league.id}/scoreboard`);
                        const data = await res.json();
                        return (data.events || []).map((e: ESPNEvent) => ({
                            ...e,
                            leagueName: league.name,
                            leagueId: league.id
                        }));
                    })
                );

                events = results
                    .filter((r): r is PromiseFulfilledResult<ESPNEvent[]> => r.status === 'fulfilled')
                    .flatMap(r => r.value);

                events.sort((a, b) => {
                    const aLive = a.status?.type?.name?.includes('IN_PROGRESS');
                    const bLive = b.status?.type?.name?.includes('IN_PROGRESS');
                    if (aLive && !bLive) return -1;
                    if (!aLive && bLive) return 1;
                    return new Date(a.date).getTime() - new Date(b.date).getTime();
                });
            } else {
                const res = await fetch(`${ESPN_BASE}${endpoint}`);
                const data = await res.json();
                events = data.events || [];
            }

            const transformed = events
                .map(transformEvent)
                .filter((g): g is Game => g !== null);

            setGames(transformed);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load games');
            setGames([]);
        } finally {
            setLoading(false);
        }
    }, [sport, endpoint]);

    useEffect(() => {
        fetchGames();
    }, [fetchGames]);

    const liveCount = games.filter(g => g.status.isLive).length;

    return { games, loading, error, refetch: fetchGames, liveCount };
}
