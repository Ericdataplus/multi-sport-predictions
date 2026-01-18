import { useState, useEffect, useCallback } from 'react';
import type { PredictionHistory } from '../types';

const STORAGE_KEY = 'prediction_history';

/**
 * Hook for prediction history in localStorage
 */
export function useHistory() {
    const [history, setHistory] = useState<PredictionHistory[]>([]);

    useEffect(() => {
        try {
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved) setHistory(JSON.parse(saved));
        } catch {
            console.error('Failed to load history');
        }
    }, []);

    useEffect(() => {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
        } catch {
            console.error('Failed to save history');
        }
    }, [history]);

    const addPrediction = useCallback((prediction: Omit<PredictionHistory, 'timestamp' | 'result'>) => {
        setHistory(prev => {
            const exists = prev.find(
                h => h.gameId === prediction.gameId && h.betType === prediction.betType
            );
            if (exists) return prev;

            return [...prev, {
                ...prediction,
                timestamp: new Date().toISOString(),
                result: 'pending' as const
            }];
        });
    }, []);

    const updateResult = useCallback((gameId: string, betType: string, result: 'win' | 'loss' | 'push') => {
        setHistory(prev => prev.map(h => {
            if (h.gameId === gameId && h.betType === betType && h.result === 'pending') {
                return { ...h, result, resolvedAt: new Date().toISOString() };
            }
            return h;
        }));
    }, []);

    const getStats = useCallback(() => {
        const resolved = history.filter(h => h.result !== 'pending');
        const wins = resolved.filter(h => h.result === 'win').length;
        const losses = resolved.filter(h => h.result === 'loss').length;
        const total = wins + losses;

        return {
            total: history.length,
            wins,
            losses,
            pending: history.filter(h => h.result === 'pending').length,
            winRate: total > 0 ? ((wins / total) * 100).toFixed(1) : '0'
        };
    }, [history]);

    return { history, addPrediction, updateResult, getStats };
}
