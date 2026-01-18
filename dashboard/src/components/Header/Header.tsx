import './Header.scss';
import { MODEL_DATA } from '../../config/constants';

export default function Header() {
    // Calculate average accuracy across all models
    const calculateStats = () => {
        let totalAcc = 0, count = 0;
        Object.values(MODEL_DATA).forEach(sport => {
            Object.values(sport).forEach(acc => {
                totalAcc += acc;
                count++;
            });
        });
        const avgAcc = count > 0 ? totalAcc / count : 0.55;
        const roi = ((avgAcc - 0.524) / 0.524 * 100).toFixed(1);
        return { accuracy: (avgAcc * 100).toFixed(0), roi };
    };

    const { accuracy, roi } = calculateStats();

    return (
        <header className="header">
            <h1>🏆 Multi-Sport Predictions</h1>
            <div className="header-stats">
                <div className="stat-item" title="AI accuracy across all sports">
                    <div className="stat-value">{accuracy}%</div>
                    <div className="stat-label">Accuracy</div>
                </div>
                <div className="stat-item" title="Return on Investment">
                    <div className="stat-value">+{roi}%</div>
                    <div className="stat-label">ROI</div>
                </div>
                <div className="stat-item" title="Trained AI models">
                    <div className="stat-value">14</div>
                    <div className="stat-label">Models</div>
                </div>
            </div>
        </header>
    );
}
