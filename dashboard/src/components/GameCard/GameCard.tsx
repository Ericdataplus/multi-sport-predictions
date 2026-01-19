import { memo } from 'react';
import './GameCard.scss';
import { formatGameTime } from '../../utils/helpers';
import type { GameCardProps } from '../../types';

// memo() = Only re-render if props actually change
const GameCard = memo(function GameCard({ game, prediction, onTrack }: GameCardProps) {
    const { status, homeTeam, awayTeam, venue, broadcast, odds, overUnder } = game;

    const homeWins = status.isFinal && homeTeam.score > awayTeam.score;
    const awayWins = status.isFinal && awayTeam.score > homeTeam.score;



    const handleTrack = () => {
        if (!status.isFinal && prediction) {
            onTrack(game.id, prediction.pick, prediction.confidence);
        }
    };

    return (
        <div className={`game-card ${status.isLive ? 'live' : ''} ${status.isFinal ? 'final' : ''}`}>
            {/* Status Row */}
            <div className="game-status">
                {status.isLive ? (
                    <span className="status-live">LIVE • Q{status.period} {status.clock}</span>
                ) : status.isFinal ? (
                    <span className="status-final">✓ FINAL</span>
                ) : (
                    <span className="status-scheduled">📅 {formatGameTime(game.date)}</span>
                )}
                <span className="broadcast">
                    {game.leagueName && `${game.leagueName} • `}
                    {broadcast && `📺 ${broadcast}`}
                </span>
            </div>

            {/* Venue & Odds */}
            {(venue || odds) && (
                <div className="venue-odds">
                    <span>🏟️ {venue}</span>
                    <span className="odds">{odds} {overUnder && `O/U ${overUnder}`}</span>
                </div>
            )}

            {/* Teams */}
            <div className="teams-container">
                <div className="team-row">
                    <div className="team-info">
                        <div className="team-logo">{awayTeam.abbreviation.substring(0, 3)}</div>
                        <div>
                            <div className="team-name">{awayTeam.name}</div>
                            <div className="team-record">{awayTeam.record}</div>
                        </div>
                    </div>
                    <div className="score-section">
                        {awayTeam.lineScores.length > 0 && (
                            <div className="line-scores">
                                {awayTeam.lineScores.map((score, i) => (
                                    <span key={i} className="quarter-score">{score}</span>
                                ))}
                            </div>
                        )}
                        <div className={`team-score ${awayWins ? 'winner' : ''} ${homeWins ? 'loser' : ''}`}>
                            {(status.isLive || status.isFinal) ? awayTeam.score : ''}
                        </div>
                    </div>
                </div>

                <div className="team-row">
                    <div className="team-info">
                        <div className="team-logo">{homeTeam.abbreviation.substring(0, 3)}</div>
                        <div>
                            <div className="team-name">{homeTeam.name}</div>
                            <div className="team-record">{homeTeam.record}</div>
                        </div>
                    </div>
                    <div className="score-section">
                        {homeTeam.lineScores.length > 0 && (
                            <div className="line-scores">
                                {homeTeam.lineScores.map((score, i) => (
                                    <span key={i} className="quarter-score">{score}</span>
                                ))}
                            </div>
                        )}
                        <div className={`team-score ${homeWins ? 'winner' : ''} ${awayWins ? 'loser' : ''}`}>
                            {(status.isLive || status.isFinal) ? homeTeam.score : ''}
                        </div>
                    </div>
                </div>
            </div>

            {/* Prediction */}
            {prediction ? (
                !status.isFinal ? (
                    <div className="game-prediction" onClick={handleTrack}>
                        <span>
                            🤖 AI Pick: <strong>{prediction.pick}</strong>
                            <span className="model-badge">V6 MODEL</span>
                        </span>
                        <span className="prediction-badge">{(prediction.confidence * 100).toFixed(0)}%</span>
                        <span className="track-icon">📌</span>
                    </div>
                ) : (
                    <div className={`game-prediction result ${prediction.pick_home === homeWins ? 'won' : 'lost'}`}>
                        <span>
                            {prediction.pick_home === homeWins ? '✅' : '❌'}
                            Picked <strong>{prediction.pick}</strong> ({(prediction.confidence * 100).toFixed(0)}%)
                        </span>
                        <span className="result-text">
                            {prediction.pick_home === homeWins ? 'WON' : 'LOST'}
                        </span>
                    </div>
                )
            ) : (
                <div className="game-prediction empty">
                    <span style={{ opacity: 0.5 }}>No Model Prediction</span>
                </div>
            )}
        </div>
    );
});

export default GameCard;
