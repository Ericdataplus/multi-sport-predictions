import { useState, useEffect } from 'react';
import rawData from '../../../data/player_props_predictions.json';

export interface PlayerProp {
    player: string;
    team: string;
    prop: string;
    line: number;
    pick: string;
    confidence: number;
    trend: string;
    player_avg: number;
    event_group?: string;
    matchup?: string;
    sport?: string;
    position?: string; // Added position
    model_accuracy?: number;
}

export function usePlayerProps(sport: string) {
    const [allProps, setAllProps] = useState<Record<string, PlayerProp[]>>({});
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        try {
            const data = rawData as any;
            const mapped: Record<string, PlayerProp[]> = {};

            if (data.sports) {
                for (const s in data.sports) {
                    mapped[s] = data.sports[s].predictions || [];
                }
            }
            setAllProps(mapped);
        } catch (e) {
            console.error("Error parsing player props data", e);
        } finally {
            setLoading(false);
        }
    }, []);

    const propsForSport = allProps[sport] || [];
    return { props: propsForSport, loading };
}
