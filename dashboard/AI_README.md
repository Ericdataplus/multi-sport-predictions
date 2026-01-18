# 🤖 AI Developer Guide - Multi-Sport Predictions Dashboard

> **READ THIS FIRST** - This document helps AI assistants understand and modify the dashboard codebase.

## 🛠️ Tech Stack
- **Vite** - Fast build tool
- **React 18** - UI framework  
- **TypeScript** - Type safety
- **SCSS** - Styles with variables/mixins

---

## 📁 Project Structure

```
dashboard/
├── AI_README.md              ← YOU ARE HERE
├── index.html                ← HTML entry point
├── vite.config.ts            ← Vite configuration
├── tsconfig.json             ← TypeScript config
│
├── src/
│   ├── main.tsx              ← React app entry
│   ├── App.tsx               ← Main app component
│   │
│   ├── types/
│   │   └── index.ts          ← TypeScript interfaces
│   │
│   ├── config/
│   │   └── constants.ts      ← Model data, sports, API URLs
│   │
│   ├── hooks/
│   │   ├── useGames.ts       ← Fetch games from ESPN
│   │   ├── usePredictions.ts ← Load predictions JSON
│   │   └── useHistory.ts     ← localStorage history
│   │
│   ├── components/
│   │   ├── Header/
│   │   │   ├── Header.tsx
│   │   │   └── Header.scss
│   │   ├── Nav/
│   │   │   ├── Nav.tsx
│   │   │   └── Nav.scss
│   │   ├── GameCard/
│   │   │   ├── GameCard.tsx
│   │   │   └── GameCard.scss
│   │   ├── GamesGrid/
│   │   │   ├── GamesGrid.tsx
│   │   │   └── GamesGrid.scss
│   │   ├── Sidebar/
│   │   │   ├── Sidebar.tsx
│   │   │   └── Sidebar.scss
│   │   ├── Picks/
│   │   │   ├── Picks.tsx
│   │   │   └── Picks.scss
│   │   ├── Parlays/
│   │   │   ├── Parlays.tsx
│   │   │   └── Parlays.scss
│   │   └── Footer/
│   │       ├── Footer.tsx
│   │       └── Footer.scss
│   │
│   ├── utils/
│   │   └── helpers.ts        ← Utility functions
│   │
│   └── styles/
│       ├── main.scss         ← Main entry
│       ├── _variables.scss   ← Colors, breakpoints
│       └── _mixins.scss      ← Reusable patterns
```

---

## 🎯 What This Dashboard Does

A **sports betting predictions dashboard** that:
1. Fetches live game data from ESPN API
2. Displays AI predictions for game outcomes
3. Supports bet types: Moneyline, Spread, O/U, Contracts
4. Tracks prediction history in localStorage
5. Auto-generates parlay suggestions
6. Supports 8 sports: NBA, NCAA, NFL, CFB, NHL, MLB, Tennis, Soccer

---

## 📊 Key TypeScript Types

Located in `src/types/index.ts`:

```typescript
interface Game {
  id: string;
  date: string;
  status: GameStatus;
  competitions: Competition[];
  leagueName?: string;
}

interface Team {
  name: string;
  abbreviation: string;
  score: number;
  isHome: boolean;
  record: string;
}

interface Prediction {
  gameId: string;
  pick: string;
  confidence: number;
  pickHome: boolean;
}
```

---

## 🎨 SCSS Variables

Located in `src/styles/_variables.scss`:

```scss
$bg-primary: #0a0a0f;
$accent-primary: #6366f1;
$accent-green: #10b981;
$accent-red: #ef4444;
$tablet: 768px;
$phone: 480px;
```

---

## ⚡ Quick Commands

```bash
# Start dev server
npm run dev

# Build for production  
npm run build

# Type check
npm run tsc
```

---

## 🔧 Common Tasks

### Add a new sport:
1. `config/constants.ts` → add to `SPORT_TABS`
2. `components/Nav/Nav.tsx` → auto-renders from config

### Add a new component:
1. Create folder: `components/NewComponent/`
2. Create `NewComponent.tsx` and `NewComponent.scss`
3. Export from component, import in parent

### Change styles:
1. Edit component's `.scss` file
2. Use variables from `_variables.scss`
3. Use mixins from `_mixins.scss`

---

## 📝 Notes for AI Assistants

1. **TypeScript** - All files use `.ts`/`.tsx` extensions
2. **Types are centralized** in `types/index.ts`
3. **Each component has co-located SCSS**
4. **Use existing SCSS variables** for consistency
5. **ESPN API is read-only** - no auth needed
6. **Predictions** come from `../data/predictions.json`
