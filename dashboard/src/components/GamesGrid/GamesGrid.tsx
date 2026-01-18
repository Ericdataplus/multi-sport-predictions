import './GamesGrid.scss';
import GameCard from '../GameCard/GameCard';
import type { Game, Prediction } from '../../types';

interface GamesGridProps {
    games: Game[];
    sport: string;
    loading: boolean;
    error: string | null;
    getPrediction: (gameId: string, sport: string) => Prediction | null;
    onTrack: (gameId: string, pick: string, confidence: number) => void;
    onRefresh: () => void;
}

export default function GamesGrid({
    games, loading, error, sport, getPrediction, onTrack, onRefresh
}: GamesGridProps) {
    if (loading) {
        return (
            <div className="loading">
                <div className="loading-spinner"></div>
                Loading games...
            </div>
        );
    }

    if (error) {
        return <div className="error">Error: {error}</div>;
    }

    if (games.length === 0) {
        return <div className="no-games">No games scheduled today</div>;
    }

    return (
        <section className="section">
            <div className="section-header">
                <h2 className="section-title">Games Today</h2>
                <button className="btn" onClick={onRefresh}>🔄 Refresh</button>
            </div>
            <div className="games-grid">
                {games.map(game => (
                    <GameCard
                        key={game.id}
                        game={game}
                        sport={sport}
                        prediction={getPrediction(game.id, sport)}
                        onTrack={onTrack}
                    />
                ))}
            </div>
        </section>
    );
}
