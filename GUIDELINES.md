# Stock Dashboard - Architecture Guidelines

This document outlines the core architectural principles, patterns, and constraints of the Stock Dashboard project. Follow these guidelines to maintain consistency and avoid breaking the established structure.

---

## Core Philosophy: SQLite First, API Second

**Historical price data is immutable.** Once a stock's price for a given date/time is recorded, it will never change. This fundamental truth drives our caching strategy:

1. **Always check SQLite before making API calls**
2. **Store fetched data immediately in SQLite**
3. **Only fetch what's missing (gap-fill strategy)**

This dramatically reduces API calls and improves startup performance, especially when the app has been closed and reopened.

---

## Data Categories

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA CATEGORIES                              │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ Immutable       │ Semi-Stale      │ Real-Time                   │
│ (SQLite forever)│ (SQLite 15min)  │ (Never cached)              │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ Daily history   │ Fundamentals    │ Current price               │
│ Intraday history│ ROIC, ROE, ROA  │ Change %                    │
│ (once fetched)  │ Sector/Industry │ Market state                │
│                 │ Options sentiment│ Pre/post market price      │
│                 │ Call/put ratio  │                             │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

### Immutable Data (SQLite Forever)
- **Daily price history** (`PriceHistory` model)
- **Intraday price history** (`IntradayPriceHistory` model)
- Once stored, never re-fetch for the same date/timestamp
- Gap-fill only: detect missing ranges, fetch only those

### Semi-Stale Data (SQLite with 15-minute TTL)
- **Stock analysis** (`StockAnalysisCache` model)
- Fundamentals, options sentiment, call/put ratios
- Check `fetched_at` timestamp before returning cached data
- Refresh if older than 15 minutes

### Real-Time Data (Never Cached)
- Current stock prices
- Price change percentages
- Market state (open/closed/pre/post)
- These are fetched fresh via the batch prices endpoint

---

## Yahoo Finance API Constraints

### Rate Limiting (The #1 Problem)

Yahoo Finance is an unofficial API with aggressive rate limiting:
- **HTTP 429** "Too Many Requests" is common under load
- **HTTP 401** "Unauthorized" can also indicate rate limiting
- Multiple simultaneous threads trigger bot detection

**Solution:** Exponential backoff with proper retry logic (see `retry.py`):
```python
RATE_LIMIT_INDICATORS = (
    "rate limit", "too many requests", "429", 
    "throttl", "exceeded", "try again", "unauthorized"
)
```

### yfinance Thread Safety Issues

yfinance is **NOT thread-safe** in the traditional sense. The real issues are:
1. **SQLite cache conflicts** - Multiple threads writing to the same cache file
2. **Rate limiting amplification** - Many threads = many simultaneous requests = ban

**Solution:** Shared cached session for all yfinance calls:
```python
# In stock_fetcher.py
_yf_session = requests_cache.CachedSession(
    'data/yfinance_http_cache',
    backend='sqlite',
    expire_after=60
)
_yf_session.headers['User-Agent'] = 'StockDashboard/1.0'

# Pass to all yfinance calls:
yf_ticker = yf.Ticker(ticker, session=get_yf_session())
```

### ThreadPoolExecutor Worker Limits

Keep worker counts **low** (4 or fewer) to naturally throttle request rates:
```python
# In config.py
yfinance_max_workers: int = 4  # NOT 10 or higher
```

---

## Batching Strategies

### When to Batch

| Library     | Batching Support | Use For                        |
|-------------|------------------|--------------------------------|
| yahooquery  | ✅ Native        | Prices, daily history, fundamentals |
| yfinance    | ✅ via `yf.download()` | Intraday history with prepost  |

### yahooquery Batching
```python
# Single API call for multiple tickers
t = YQTicker(['AAPL', 'MSFT', 'GOOG'])
all_prices = t.price  # One call, multiple results
all_history = t.history(start=..., end=...)  # Batched history
```

### yfinance Batching (use `yf.download()`)
```python
# Built-in threading, handles rate limits
data = yf.download(
    tickers=['AAPL', 'MSFT', 'GOOG'],
    period='5d',
    interval='5m',
    prepost=True,
    group_by='ticker',
    threads=True,
    session=get_yf_session()
)
```

**⚠️ Never write your own threading loop for yfinance** - use `yf.download()` which handles this safely.

### When NOT to Batch
- Single ticker lookups for specific user actions (e.g., adding a new holding)
- Real-time price polling (already optimized with yahooquery batch)

---

## Gap-Fill Strategy

When the app starts after being closed, there may be missing data. The gap-fill strategy:

1. **Analyze coverage** - What data do we have? What's missing?
2. **Identify gaps** - Missing start range? Missing end range? Internal gaps?
3. **Batch fetch only gaps** - Don't re-fetch existing data
4. **Store immediately** - Write to SQLite as soon as data arrives

```python
# Example from price_history.py
def analyze_intraday_coverage(self, ticker, interval, start, end):
    stored = self.get_intraday_history(...)
    if not stored:
        return {"needs_fetch": True, "missing_ranges": [(start, end)]}
    
    # Check for gaps...
    return {
        "needs_fetch": len(missing_ranges) > 0,
        "missing_ranges": missing_ranges,
        "stored_data": stored
    }
```

---

## Service Architecture

### Service Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Endpoints                         │
│                         (main.py)                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│ StockFetcher  │  │ StockAnalysis   │  │ OptionPricing       │
│ (prices,      │  │ (fundamentals,  │  │ (option chains,     │
│  history)     │  │  options data)  │  │  Greeks)            │
└───────┬───────┘  └────────┬────────┘  └──────────┬──────────┘
        │                   │                      │
        ▼                   ▼                      ▼
┌───────────────────────────────────────────────────────────────┐
│                    PriceHistoryService                         │
│                 (SQLite read/write layer)                      │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                     SQLite Database                            │
│  - PriceHistory (daily)                                        │
│  - IntradayPriceHistory                                        │
│  - StockAnalysisCache                                          │
└───────────────────────────────────────────────────────────────┘
```

### Singleton Pattern

All services use singletons to share state (executors, sessions, caches):
```python
_stock_fetcher: Optional[StockFetcher] = None

def get_stock_fetcher() -> StockFetcher:
    global _stock_fetcher
    if _stock_fetcher is None:
        _stock_fetcher = StockFetcher()
    return _stock_fetcher
```

---

## Frontend Patterns

### Batch-First Approach

The frontend should always prefer batch endpoints:
```typescript
// ✅ Good - one API call for all tickers
const historyData = await api.getBatchHistory(tickers, period);

// ❌ Bad - N API calls for N tickers
for (const ticker of tickers) {
  await api.getStockHistory(ticker, period);
}
```

### Data Flow: Prices vs History

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend Data Flow                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  App Start                                                       │
│      │                                                           │
│      ├──► Lite Portfolio Load (instant, no prices)              │
│      │                                                           │
│      ├──► Batch History Fetch ──► SQLite (gap-fill only)        │
│      │                                                           │
│      └──► Price Refresh Loop (every 3 seconds)                  │
│               │                                                  │
│               └──► Batch Prices ──► Update UI + Chart Points    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Chart Updates

- **1d period**: Real-time updates via `chart_point` in batch price response
- **Other periods**: Full history from batch endpoint, no real-time updates needed

---

## Adding New Features - Checklist

When adding new data fetching features:

1. **Is this data immutable?**
   - Yes → Store in SQLite forever, never re-fetch
   - No → Consider TTL-based caching (like `StockAnalysisCache`)

2. **Can this be batched?**
   - Yes → Use yahooquery for multiple tickers, or `yf.download()`
   - No → Use retry wrapper with exponential backoff

3. **Does this use yfinance?**
   - Yes → Use the shared session: `yf.Ticker(ticker, session=get_yf_session())`
   - Yes → Keep thread pool workers low (≤4)

4. **Is this user-facing latency sensitive?**
   - Yes → Check SQLite first, return cached if valid
   - Yes → Consider separate "fast path" endpoint

5. **Could this hit rate limits?**
   - Yes → Wrap in `make_yahoo_request()` for retry logic
   - Yes → Never add fixed `time.sleep()` delays

---

## Anti-Patterns to Avoid

### ❌ Fixed Delays
```python
# Bad - wastes time when not rate limited
time.sleep(2.0)

# Good - exponential backoff handles this
result = make_yahoo_request(lambda: ..., description="...")
```

### ❌ Multiple YQTicker Instances
```python
# Bad - 4 separate instances, 4x the overhead
t1 = YQTicker(ticker)
stats = t1.key_stats
t2 = YQTicker(ticker)  # Why create another?
profile = t2.asset_profile

# Good - single instance, reuse for all properties
t = YQTicker(ticker)
stats = t.key_stats
profile = t.asset_profile
```

### ❌ Sequential Fetches When Batching is Available
```python
# Bad - N API calls
for ticker in tickers:
    data = YQTicker(ticker).price

# Good - 1 API call
t = YQTicker(tickers)
all_data = t.price
```

### ❌ Ignoring SQLite Cache
```python
# Bad - always fetches from API
def get_history(ticker, start, end):
    return yf.Ticker(ticker).history(start=start, end=end)

# Good - check cache first
def get_history(ticker, start, end):
    cached = price_history.get_daily_history(ticker, start, end)
    if has_complete_coverage(cached, start, end):
        return cached
    # Only fetch missing ranges...
```

---

## Environment & Configuration

Key settings in `config.py`:

```python
# Threading
yfinance_max_workers: int = 4  # Keep low to avoid rate limits

# Retry logic
yahoo_max_retries: int = 5
yahoo_base_delay: float = 1.0   # Initial backoff delay
yahoo_max_delay: float = 32.0   # Cap on backoff delay

# Caching
cache_ttl_seconds: int = 60     # In-memory cache for stock data
data_retention_days: int = 730  # SQLite cleanup threshold
```

---

## Summary

1. **SQLite is the source of truth** for historical data
2. **Batch everything** that can be batched
3. **Share the yfinance session** across all calls
4. **Respect rate limits** with exponential backoff
5. **Never add fixed delays** - let the retry logic handle it
6. **Keep thread workers low** (≤4) to avoid triggering rate limits
7. **Gap-fill only** - never re-fetch data you already have

