import './Nav.scss';
import { SPORT_TABS } from '../../config/constants';
import type { NavProps } from '../../types';

export default function Nav({ currentSport, onSportChange, liveCount }: NavProps) {
    return (
        <nav className="nav-tabs">
            {SPORT_TABS.map(tab => (
                <button
                    key={tab.id}
                    className={`nav-tab ${currentSport === tab.id ? 'active' : ''}`}
                    onClick={() => onSportChange(tab.id, tab.endpoint)}
                >
                    {tab.label}
                    {currentSport === tab.id && liveCount > 0 && (
                        <span className="live-badge">{liveCount} LIVE</span>
                    )}
                </button>
            ))}
        </nav>
    );
}
