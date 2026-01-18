import React, { useState, memo } from 'react';
import './Picks.scss';
import { BET_TYPES, BET_EXPLAINERS, MODEL_DATA } from '../../config/constants';
import { seededRandom } from '../../utils/helpers';
import { usePlayerProps } from '../../hooks/usePlayerProps';
import type { PicksProps } from '../../types';

const Picks = memo(function Picks({ sport, pickType, onPickTypeChange }: PicksProps) {
    const [searchTerm, setSearchTerm] = useState('');
    const sportAccuracy = MODEL_DATA[sport] || { moneyline: 0.55 };

    // Load Real Player Props
    const { props: playerProps, loading: propsLoading } = usePlayerProps(sport);

    // Generate sample picks for other types (Mock)
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

    // Prepare Props for Display
    const getDisplayProps = () => {
        if (pickType !== 'props') return [];
        if (!playerProps) return [];

        const groupPriority: Record<string, number> = {
            'National Championship': 10,
            'AFC Championship': 5,
            'NFC Championship': 5,
            'Playoff Matchup': 4,
            'Regular Season': 0
        };

        let activeProps = [...playerProps];

        // Filter
        if (searchTerm) {
            const lower = searchTerm.toLowerCase();
            activeProps = activeProps.filter(p =>
                p.player.toLowerCase().includes(lower) ||
                (p.team && p.team.toLowerCase().includes(lower)) ||
                (p.prop && p.prop.toLowerCase().includes(lower)) ||
                (p.position && p.position.toLowerCase().includes(lower))
            );
        }

        return activeProps.sort((a, b) => {
            const gA = a.event_group || 'Regular Season';
            const gB = b.event_group || 'Regular Season';
            const pA = groupPriority[gA] || 0;
            const pB = groupPriority[gB] || 0;
            if (pA !== pB) return pB - pA;
            return b.confidence - a.confidence;
        }).slice(0, 50); // Limit to top 50
    };

    const mockPicks = (pickType !== 'history' && pickType !== 'contracts' && pickType !== 'props') ? generatePicks() : [];
    const realProps = getDisplayProps();

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

            {/* Search Bar for Props */}
            {pickType === 'props' && (
                <div style={{ marginBottom: '1rem', marginTop: '0.5rem' }}>
                    <input
                        type="text"
                        placeholder="🔍 Search player, team, or position..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        style={{
                            width: '100%',
                            padding: '0.6rem 0.8rem',
                            borderRadius: '0.5rem',
                            border: '1px solid #374151',
                            backgroundColor: '#1f2937',
                            color: '#f3f4f6',
                            fontSize: '0.9rem',
                            outline: 'none'
                        }}
                    />
                </div>
            )}

            {/* Loading State for Props */}
            {pickType === 'props' && propsLoading && <div className="loading">Loading Props...</div>}

            {/* Picks List (Mock) */}
            <div className="picks-list">
                {mockPicks.map((pick, idx) => (
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

                {/* Real Player Props List */}
                {pickType === 'props' && realProps.map((prop, idx) => {
                    const group = prop.event_group || 'Regular Season';
                    const prevGroup = idx > 0 ? (realProps[idx - 1].event_group || 'Regular Season') : null;
                    const showHeader = group !== 'Regular Season' && group !== prevGroup;

                    return (
                        <div key={idx} className="prop-wrapper">
                            {showHeader && (
                                <div style={{
                                    fontSize: '0.8rem',
                                    fontWeight: 700,
                                    color: '#facc15',
                                    margin: '0.75rem 0 0.4rem 0.2rem',
                                    borderBottom: '1px solid #374151',
                                    paddingBottom: '0.2rem',
                                    display: 'flex',
                                    justifyContent: 'space-between'
                                }}>
                                    <span>{group}</span>
                                    <span style={{ fontWeight: 400, opacity: 0.8, fontSize: '0.7rem' }}>{prop.matchup}</span>
                                </div>
                            )}
                            <div className="pick-card">
                                <div className="pick-info">
                                    <div className="pick-team" style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                                        {prop.player}
                                        <span style={{ fontSize: '0.7rem', color: '#9ca3af', fontWeight: 400 }}>
                                            ({prop.position || 'Unknown'} - {prop.team})
                                        </span>
                                    </div>
                                    <div className="pick-type">{prop.prop}: {prop.line !== null ? prop.line : ''} {prop.pick}</div>
                                </div>
                                <div className="pick-confidence">
                                    <div className="confidence-value">{(prop.confidence * 100).toFixed(0)}%</div>
                                    <div className="ev-badge" style={{ fontSize: '0.6rem', color: '#9ca3af' }}>
                                        Avg: {prop.player_avg}
                                    </div>
                                </div>
                            </div>
                        </div>
                    );
                })}

                {/* Empty State if search yields no results */}
                {pickType === 'props' && !propsLoading && realProps.length === 0 && (
                    <div style={{ textAlign: 'center', padding: '2rem', color: '#9ca3af', fontSize: '0.9rem' }}>
                        {searchTerm ? 'No players found matching your search.' : 'No props available for this sport.'}
                    </div>
                )}

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
