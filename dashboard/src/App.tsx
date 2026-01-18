import { useState, useEffect, useCallback } from 'react';
import './styles/main.scss';

import Header from './components/Header/Header';
import Nav from './components/Nav/Nav';
import GamesGrid from './components/GamesGrid/GamesGrid';
import Sidebar from './components/Sidebar/Sidebar';
import Picks from './components/Picks/Picks';
import Parlays from './components/Parlays/Parlays';
import Footer from './components/Footer/Footer';

import { useGames } from './hooks/useGames';
import { usePredictions } from './hooks/usePredictions';
import { useHistory } from './hooks/useHistory';
import { REFRESH_INTERVAL } from './config/constants';

function App() {
  const [currentSport, setCurrentSport] = useState('nba');
  const [currentEndpoint, setCurrentEndpoint] = useState('/basketball/nba/scoreboard');
  const [pickType, setPickType] = useState('moneyline');

  // Hooks
  const { games, loading, error, refetch, liveCount } = useGames(currentSport, currentEndpoint);
  const { getPrediction } = usePredictions();
  const { addPrediction } = useHistory();

  // Sport change handler
  const handleSportChange = useCallback((sport: string, endpoint: string) => {
    setCurrentSport(sport);
    setCurrentEndpoint(endpoint);
  }, []);

  // Track prediction
  const handleTrack = useCallback((gameId: string, pick: string, confidence: number) => {
    addPrediction({
      gameId,
      sport: currentSport,
      betType: 'moneyline',
      pick,
      confidence,
      odds: '-'
    });
  }, [addPrediction, currentSport]);

  // Auto refresh
  useEffect(() => {
    const interval = setInterval(refetch, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [refetch]);

  return (
    <div className="app">
      <Header />

      <Nav
        currentSport={currentSport}
        onSportChange={handleSportChange}
        liveCount={liveCount}
      />

      <main className="main-content">
        <GamesGrid
          games={games}
          sport={currentSport}
          loading={loading}
          error={error}
          getPrediction={getPrediction}
          onTrack={handleTrack}
          onRefresh={refetch}
        />

        <Sidebar>
          <Picks
            sport={currentSport}
            pickType={pickType}
            onPickTypeChange={setPickType}
          />

          <Parlays
            games={games}
            sport={currentSport}
            predictions={{}}
          />
        </Sidebar>
      </main>

      <Footer />
    </div>
  );
}

export default App;
