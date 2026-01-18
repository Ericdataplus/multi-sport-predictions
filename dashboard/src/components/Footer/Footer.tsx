import './Footer.scss';

export default function Footer() {
    return (
        <footer className="footer">
            Last updated: {new Date().toLocaleTimeString()} |
            V6 Models trained: 12/23/2025 |
            <a href="#">View Training Results</a>
        </footer>
    );
}
