import { useState, useEffect } from 'react';
import type { Prediction } from '../types';

/**
 * Hook to load predictions from JSON
 */
export function usePredictions() {
    const [predictions, setPredictions] = useState<Record<string, Prediction[]>>({});
    const [loaded, setLoaded] = useState(false);

    useEffect(() => {
        async function load() {
            try {
                const res = await fetch('../data/predictions.json');
                const data = await res.json();
                setPredictions(data.predictions || {});
                setLoaded(true);
            } catch {
                console.warn('Could not load predictions.json');
                setLoaded(false);
            }
        }
        load();
    }, []);

    const getPrediction = (gameId: string, sport: string): Prediction | null => {
        if (!loaded || !predictions[sport]) return null;
        return predictions[sport].find(p => p.game_id === gameId) || null;
    };

    return { predictions, loaded, getPrediction };
}
