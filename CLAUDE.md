# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

### Full Stack (Docker)
```bash
docker-compose up --build
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# SQLite Browser: http://localhost:8001
```

### Backend (Manual)
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend (Manual)
```bash
cd frontend
npm install
npm run dev      # Dev server with hot reload
npm run build    # Production build
```

### Testing
```bash
cd backend
pytest                              # All tests
pytest -v tests/test_api.py         # Single file
pytest --cov                        # With coverage
```

## Architecture Overview

### Core Philosophy: SQLite First, API Second
Historical price data is immutable. Once stored, never re-fetch. This drives all caching decisions.

**Data Categories:**
- **Immutable (SQLite forever):** Daily history (`PriceHistory`), intraday history (`IntradayPriceHistory`)
- **Semi-Stale (15-min TTL):** Fundamentals, options data (`StockAnalysisCache`)
- **Real-Time (Never cached):** Current prices, market state

### Backend Service Stack
```
FastAPI Endpoints (main.py)
    ↓
Service Layer (app/services/)
├── StockFetcher       - Hybrid yahooquery + yfinance data fetching
├── PriceHistoryService - SQLite cache layer for price data
├── StockAnalysisService - Fundamentals + options aggregation
├── WFOOptimizer       - Walk-Forward Optimization engine
└── CalibrationService - Long-running calibration job management
    ↓
SQLAlchemy Models (models.py) → SQLite
```

Services use singleton pattern via `app/services/__init__.py` getters.

### Frontend Component Stack
```
src/
├── components/        - Reusable UI (HoldingCard, StockAnalysisModal)
│   └── holding/       - Holding sub-components (ActionScoreBadge, MiniStockChart)
├── pages/             - Full layouts (CalibrationPage)
├── hooks/             - usePortfolio, useStockAnalysis, useDataCache
├── services/          - API clients (api.ts), scoring logic (stockScoring.ts)
├── context/           - DataCacheContext, ViewContext (swing vs trend view)
└── types/             - TypeScript interfaces
```

### Walk-Forward Optimization (WFO)
The WFO engine calibrates indicator weights per stock for two horizons:
- **Swing (3 days):** Short-term mean reversion
- **Trend (15 days):** Trend following

Key files:
- `backend/app/services/wfo_optimizer.py` - Two-Pass Coordinate Descent optimization
- `frontend/src/services/stockScoring.ts` - Applies calibrated weights to indicators
- `frontend/src/context/ViewContext.tsx` - Global horizon state

## Critical Development Constraints

### Yahoo Finance Rate Limiting
- Keep `yfinance_max_workers ≤ 4` (in `config.py`)
- Use `make_yahoo_request()` wrapper for exponential backoff
- Always pass shared session: `yf.Ticker(ticker, session=get_yf_session())`
- Use `yf.download()` for batched history, never custom threading loops

### Batching Requirements
```python
# Good - single API call
t = YQTicker(['AAPL', 'MSFT', 'GOOG'])
all_prices = t.price

# Bad - multiple API calls
for ticker in tickers:
    data = YQTicker(ticker).price
```

### Gap-Fill Strategy
Always check SQLite before fetching. Only fetch missing date ranges:
```python
# Check cache → identify gaps → fetch only missing → store immediately
```

### Anti-Patterns to Avoid
- Fixed `time.sleep()` delays (use exponential backoff)
- Multiple YQTicker instances for same ticker
- Sequential fetches when batching is available
- Ignoring SQLite cache for historical data
- Custom threading loops for yfinance (use `yf.download()`)

## API Structure

Versioned API (`/api/v1/`):
- `GET/PUT /portfolio` - Portfolio CRUD
- `POST/DELETE /holdings` - Holdings management
- `GET /stock/{ticker}` - Detailed stock data
- `POST /stocks/batch` - Batch price updates
- `POST /calibrate/start` - Start WFO calibration
- `GET /calibrate/status` - Calibration progress

## Database Models

- `Portfolio`, `Holding`, `OptionHolding` - Portfolio tracking
- `PriceHistory`, `IntradayPriceHistory` - Immutable price data
- `StockAnalysisCache` - TTL-based fundamentals cache
- `CalibrationWeights`, `CalibrationWindow`, `CalibrationTrade` - WFO results
