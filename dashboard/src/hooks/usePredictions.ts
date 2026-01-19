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
                const res = await fetch('/predictions.json');
                const data = await res.json();
                const raw = data.predictions || {};
                const processed: Record<string, Prediction[]> = {};

                // Flatten predictions to match UI interface
                Object.keys(raw).forEach(sport => {
                    processed[sport] = raw[sport].map((p: any) => ({
                        ...p,
                        pick: p.pick || p.predictions?.moneyline?.pick,
                        confidence: p.confidence || p.predictions?.moneyline?.confidence,
                        pick_home: p.pick_home ?? p.predictions?.moneyline?.pick_home
                    }));
                });

                setPredictions(processed);
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
