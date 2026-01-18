import { memo } from 'react';
import './Parlays.scss';
import { PARLAY_CONFIGS } from '../../config/constants';
import type { ParlaysProps } from '../../types';

const Parlays = memo(function Parlays({ games }: ParlaysProps) {
    if (games.length < 2) {
        return (
            <div className="section">
                <div className="section-header">
                    <h3 className="section-title">🎰 Parlay Builder</h3>
                </div>
                <div className="no-parlays">Need 2+ games for parlays</div>
            </div>
        );
    }

    return (
        <div className="section">
            <div className="section-header">
                <h3 className="section-title">🎰 Parlay Builder</h3>
            </div>

            <div className="parlay-explainer">
                💡 <strong>Parlay</strong> = Combine picks for bigger payouts. ALL must win!
            </div>

            <div className="parlays-list">
                {PARLAY_CONFIGS.filter(p => p.legs <= games.length).map((parlay, idx) => {
                    const picks = games.slice(0, parlay.legs).map(g => ({
                        team: g.homeTeam.abbreviation,
                        conf: 65
                    }));

                    return (
                        <div key={idx} className="parlay-card">
                            <div className="parlay-header">
                                <span className="parlay-legs">{parlay.legs}-Leg Parlay</span>
                                <span className="parlay-odds">{parlay.odds}</span>
                            </div>

                            <div className="parlay-picks">
                                {picks.map((pick, i) => (
                                    <div key={i} className="parlay-pick">
                                        <span>{pick.team}</span>
                                        <span>{pick.conf}%</span>
                                    </div>
                                ))}
                            </div>

                            <div className="parlay-footer">
                                <span className={`risk-badge risk-${parlay.risk}`}>
                                    {parlay.risk.toUpperCase()} RISK
                                </span>
                                <span className="parlay-payout">$100 → ${parlay.payout}</span>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
});

export default Parlays;
