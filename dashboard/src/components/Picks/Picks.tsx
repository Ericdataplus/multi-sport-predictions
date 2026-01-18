import { memo } from 'react';
import './Picks.scss';
import { BET_TYPES, BET_EXPLAINERS, MODEL_DATA } from '../../config/constants';
import { seededRandom } from '../../utils/helpers';
import type { PicksProps } from '../../types';

const Picks = memo(function Picks({ sport, pickType, onPickTypeChange }: PicksProps) {
    const sportAccuracy = MODEL_DATA[sport] || { moneyline: 0.55 };

    // Generate sample picks based on sport and type
    const generatePicks = () => {
        const teams = ['Lakers', 'Celtics', 'Warriors', 'Heat', 'Bucks', 'Suns', '76ers', 'Nuggets'];
        const picks = [];

        for (let i = 0; i < 4; i++) {
            const seed = `${sport}_${pickType}_${i}`;
            const rand1 = seededRandom(seed + '_conf');
            const rand2 = seededRandom(seed + '_team');

            const typeKey = pickType === 'total' ? 'overunder' : pickType;
            const baseAcc = sportAccuracy[typeKey] || 0.55;
            const conf = Math.min(baseAcc + (rand1 * 0.08 - 0.04), 0.88);
            const team = teams[Math.floor(rand2 * teams.length)];

            let pickText = team;
            let oddsText = conf > 0.55 ? `-${100 + Math.floor(rand1 * 50)}` : `+${100 + Math.floor(rand1 * 80)}`;

            if (pickType === 'spread') {
                const spread = (rand1 > 0.5 ? '-' : '+') + (Math.floor(rand1 * 8) + 1) + '.5';
                pickText = `${team} ${spread}`;
                oddsText = '-110';
            } else if (pickType === 'total') {
                const total = 200 + Math.floor(rand2 * 40) + '.5';
                pickText = `${rand2 > 0.5 ? 'Over' : 'Under'} ${total}`;
                oddsText = '-110';
            }

            picks.push({ team: pickText, conf, odds: oddsText, ev: ((conf - 0.524) * 10).toFixed(1) });
        }

        return picks.sort((a, b) => b.conf - a.conf);
    };

    const picks = pickType !== 'history' && pickType !== 'contracts' ? generatePicks() : [];

    return (
        <div className="section">
            <div className="section-header">
                <h3 className="section-title">🎯 Best Picks</h3>
            </div>

            {/* Bet Type Tabs */}
            <div className="picks-tabs">
                {BET_TYPES.map(type => (
                    <button
                        key={type.id}
                        className={`picks-tab ${pickType === type.id ? 'active' : ''}`}
                        onClick={() => onPickTypeChange(type.id)}
                        title={type.title}
                    >
                        {type.label}
                    </button>
                ))}
            </div>

            {/* Explainer */}
            <div
                className="bet-explainer"
                dangerouslySetInnerHTML={{ __html: BET_EXPLAINERS[pickType] || '' }}
            />

            {/* Picks List */}
            <div className="picks-list">
                {picks.map((pick, idx) => (
                    <div key={idx} className="pick-card">
                        <div className="pick-info">
                            <div className="pick-team">{pick.team}</div>
                            <div className="pick-odds">{pick.odds}</div>
                        </div>
                        <div className="pick-confidence">
                            <div className="confidence-value">{(pick.conf * 100).toFixed(0)}%</div>
                            <div className="ev-badge">+{pick.ev} EV</div>
                        </div>
                    </div>
                ))}

                {pickType === 'history' && (
                    <div className="history-placeholder">
                        <p>📜 Prediction history will appear here</p>
                        <p className="hint">Track predictions by clicking 📌 on game cards</p>
                    </div>
                )}

                {pickType === 'contracts' && (
                    <div className="contracts-guide">
                        <p className="guide-title">📈 How Contracts Work:</p>
                        <p>• Contracts priced $0.01 - $0.99 (price = probability)</p>
                        <p>• If you're RIGHT: each contract pays $1.00</p>
                        <p>• If you're WRONG: contract worth $0.00</p>
                        <p className="tip">💡 Only bet when AI confidence {'>'} market price by 5%+</p>
                    </div>
                )}
            </div>
        </div>
    );
});

export default Picks;
