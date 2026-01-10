# Stock Portfolio Dashboard with WFO

This project is a sophisticated full-stack application designed for stock portfolio tracking and technical analysis. It features a unique **Walk-Forward Optimization (WFO)** engine that calibrates technical indicator weights based on historical performance.

## Project Overview

*   **Type:** Full-Stack Web Application (Monorepo)
*   **Purpose:** Track stock holdings, view real-time metrics, and receive algorithmic trading signals optimized for specific time horizons.
*   **Key Differentiator:** The WFO engine dynamically adjusts the importance of indicators (RSI, MACD, etc.) for each stock to maximize the System Quality Number (SQN).

## Architecture

### Backend (`/backend`)
*   **Framework:** FastAPI (Python 3.11+)
*   **Database:** SQLite with SQLAlchemy (AsyncIO)
*   **Data Sources:** `yahooquery` and `yfinance` for market data.
*   **Core Services:**
    *   `wfo_optimizer.py`: Implements Two-Pass Coordinate Descent to find optimal indicator weights.
    *   `wfo_simulator.py`: Vectorized backtesting engine (Pandas/NumPy) for fast simulation.
    *   `stock_analysis.py`: Aggregates fundamentals and options data.
    *   `calibration_service.py`: Manages the long-running calibration jobs.

### Frontend (`/frontend`)
*   **Framework:** React 18 (Vite)
*   **Language:** TypeScript
*   **Styling:** Tailwind CSS (Glassmorphism design system)
*   **State Management:** React Context (`DataCacheContext`, `ViewContext`)
*   **Visualization:** Recharts for price history and radar charts.

## Key Features

1.  **Horizon-Based Analysis:**
    *   **Swing View (3 Days):** Optimized for short-term mean reversion and momentum.
    *   **Trend View (15 Days):** Optimized for trend following.
    *   Toggled via `ViewContext` in the dashboard.

2.  **Walk-Forward Optimization (WFO):**
    *   Calibrates weights on a rolling window basis.
    *   Filters out "low confidence" strategies (SQN < 3.0).
    *   Results are persisted in the database and served to the frontend.

3.  **Options Analytics:**
    *   Tracks Cash-Secured Puts (CSP) and Covered Calls (CC).
    *   Calculates Greeks and visualizes P/L.

## Building and Running

### Method 1: Docker (Recommended)
The easiest way to run the full stack:
```bash
docker-compose up --build
```
*   Frontend: `http://localhost:3000`
*   Backend API: `http://localhost:8000`

### Method 2: Manual Setup

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
# Run development server
npm run dev
```

## Development Conventions

### Backend
*   **Async First:** Use `async/await` for all route handlers and database operations.
*   **Pandas for Data:** Use Pandas DataFrames for heavy calculation (WFO, simulations) to ensure performance.
*   **Service Pattern:** Logic resides in `app/services/`, not in route handlers.
*   **Caching:** Extensive use of `aiosqlite` and in-memory caching to respect API rate limits.

### Frontend
*   **Hooks Pattern:** Logic is encapsulated in custom hooks (e.g., `useStockAnalysis`, `usePortfolio`).
*   **Context API:** Used for global state that doesn't change often (View settings, Cache instances).
*   **Component Structure:**
    *   `components/`: Reusable UI components.
    *   `pages/`: Full page layouts (`CalibrationPage`).
    *   `services/`: Pure TypeScript API clients and logic (`stockScoring.ts`).

## Key Files to Know

*   `backend/app/services/wfo_optimizer.py`: The "brain" of the calibration system.
*   `frontend/src/services/stockScoring.ts`: The scoring engine that applies weights to raw indicators.
*   `frontend/src/hooks/useStockAnalysis.ts`: The bridge that fetches data, requests analysis, and handles caching/updates.
*   `frontend/src/context/ViewContext.tsx`: Controls the global prediction horizon state.
