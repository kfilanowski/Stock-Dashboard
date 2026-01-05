# Stock Portfolio Dashboard

A modern, containerized stock portfolio dashboard built with FastAPI and React.

## Features

- **Mock Portfolio Tracking**: Set a total portfolio value and allocate percentages to different stocks
- **Live Stock Data**: Real-time prices via Yahoo Finance (yfinance)
- **YTD Performance**: Track year-to-date returns for each holding
- **SMA(200)**: View the 200-day Simple Moving Average and price comparison
- **Interactive Charts**: Visualize stock price history with Recharts
- **Auto-Refresh**: Data updates every 30 seconds
- **Dark Modern UI**: Glassmorphism design with smooth animations

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI + Python 3.11 |
| Database | SQLite with SQLAlchemy 2.0 |
| Stock Data | yfinance (free, no API key) |
| Frontend | React 18 + TypeScript |
| Styling | Tailwind CSS |
| Charts | Recharts |
| Container | Docker Compose |

## Quick Start

```bash
# Clone and start
docker-compose up --build

# Access the dashboard
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

## Usage

1. **Set Portfolio Value**: Click the gear icon to set your total investment amount (default: $10,000)
2. **Add Holdings**: Click "Add Stock" to add a ticker with an allocation percentage
3. **Track Performance**: View live prices, YTD returns, and SMA indicators
4. **View Details**: Click the chart icon on any holding to see detailed charts

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/portfolio` | Get portfolio with live stock data |
| PUT | `/api/portfolio` | Update portfolio settings |
| POST | `/api/holdings` | Add a stock holding |
| DELETE | `/api/holdings/{id}` | Remove a holding |
| GET | `/api/stock/{ticker}` | Get detailed stock data |

## Project Structure

```
Stock-Dashboard/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI routes
│   │   ├── models.py        # SQLAlchemy models
│   │   ├── schemas.py       # Pydantic schemas
│   │   ├── database.py      # DB configuration
│   │   └── stock_service.py # yfinance integration
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── hooks/           # Custom React hooks
│   │   ├── services/        # API client
│   │   └── types/           # TypeScript types
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## Data Persistence

Portfolio data is stored in SQLite and persisted via Docker volume (`sqlite_data`). Your portfolio settings and holdings will survive container restarts.

## Notes

- Stock data is delayed ~15 minutes (Yahoo Finance free tier)
- SMA(200) requires 200+ trading days of history to be accurate
- The dashboard auto-refreshes every 30 seconds

