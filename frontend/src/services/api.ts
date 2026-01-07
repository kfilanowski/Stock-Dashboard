/**
 * API service for the Stock Dashboard.
 * 
 * All API calls are centralized here for maintainability.
 * Uses versioned API endpoints (v1) with fallback support.
 */
import type { Portfolio, StockData, StockHistoryResponse } from '../types';

// API Configuration
const API_VERSION = 'v1';
const API_BASE = `/api/${API_VERSION}`;

/**
 * Handle API response and extract JSON or throw error.
 */
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || 'Request failed');
  }
  return response.json();
}

// ============================================================================
// Portfolio API
// ============================================================================

/**
 * Get the portfolio with holdings.
 * @param lite - If true, returns structure without live stock data (faster)
 */
export async function getPortfolio(lite = false): Promise<Portfolio> {
  const url = lite ? `${API_BASE}/portfolio?lite=true` : `${API_BASE}/portfolio`;
  const response = await fetch(url);
  return handleResponse<Portfolio>(response);
}

export interface UpdatePortfolioData {
  name?: string;
  total_value?: number;
  chart_period?: string;
  sort_field?: string;
  sort_direction?: string;
}

/**
 * Update portfolio settings.
 */
export async function updatePortfolio(data: UpdatePortfolioData): Promise<Portfolio> {
  const response = await fetch(`${API_BASE}/portfolio`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return handleResponse<Portfolio>(response);
}

// ============================================================================
// Holdings API
// ============================================================================

export interface AddHoldingData {
  ticker: string;
  shares?: number;
  avg_cost?: number | null;
}

/**
 * Add a new holding to the portfolio.
 */
export async function addHolding(data: AddHoldingData): Promise<void> {
  const response = await fetch(`${API_BASE}/holdings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return handleResponse<void>(response);
}

/**
 * Delete a holding from the portfolio.
 */
export async function deleteHolding(holdingId: number): Promise<void> {
  const response = await fetch(`${API_BASE}/holdings/${holdingId}`, {
    method: 'DELETE',
  });
  return handleResponse<void>(response);
}

export interface UpdateHoldingData {
  shares?: number;
  avg_cost?: number | null;
}

/**
 * Update a holding's allocation or investment info.
 */
export async function updateHolding(holdingId: number, data: UpdateHoldingData): Promise<void> {
  const response = await fetch(`${API_BASE}/holdings/${holdingId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return handleResponse<void>(response);
}

// ============================================================================
// Stock API
// ============================================================================

/**
 * Get detailed stock data including price, YTD, SMA, and history.
 */
export async function getStock(ticker: string): Promise<StockData> {
  const response = await fetch(`${API_BASE}/stock/${ticker}`);
  return handleResponse<StockData>(response);
}

export interface StockQuote {
  ticker: string;
  current_price: number;
  previous_close: number;
  change: number;
  change_pct: number;
  ytd_return: number;
  sma_200: number | null;
  price_vs_sma: number | null;
}

/**
 * Get just the current price data for a single stock (lightweight).
 */
export async function getStockQuote(ticker: string): Promise<StockQuote> {
  const response = await fetch(`${API_BASE}/stock/${ticker}/quote`);
  return handleResponse<StockQuote>(response);
}

/**
 * Get historical data for a stock.
 * @param ticker - Stock ticker symbol
 * @param period - Time period (1d, 3d, 1w, 1mo, 3mo, 6mo, ytd, 1y, 2y)
 */
export async function getStockHistory(ticker: string, period: string = '1y'): Promise<StockHistoryResponse> {
  const response = await fetch(`${API_BASE}/stock/${ticker}/history?period=${period}`);
  return handleResponse<StockHistoryResponse>(response);
}

/**
 * Clear all cached price history for a stock (forces full refresh).
 */
export async function clearStockHistory(ticker: string): Promise<void> {
  const response = await fetch(`${API_BASE}/stock/${ticker}/cache`, {
    method: 'DELETE',
  });
  return handleResponse<void>(response);
}

// ============================================================================
// Batch Prices API (Optimized for fast updates)
// ============================================================================

export interface ChartPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface BatchPriceData {
  ticker: string;
  current_price: number;
  previous_close: number;
  change: number;
  change_pct: number;
  market_state: string;
  chart_point: ChartPoint | null;
}

/**
 * Get current prices for multiple tickers in one efficient batch call.
 * 
 * This is optimized for frequent polling - only fetches price data,
 * not historical data or other metadata. Much faster than individual calls.
 * 
 * @param tickers - Array of ticker symbols
 * @returns Dict of ticker -> price data
 */
export async function getBatchPrices(tickers: string[]): Promise<Record<string, BatchPriceData>> {
  if (!tickers.length) return {};
  
  const tickerStr = tickers.join(',');
  const response = await fetch(`${API_BASE}/prices?tickers=${encodeURIComponent(tickerStr)}`);
  return handleResponse<Record<string, BatchPriceData>>(response);
}

// ============================================================================
// Stock Analysis API (Fundamentals, Options, Sector Data)
// ============================================================================

export interface StockFundamentals {
  ticker: string;
  sector: string | null;
  industry: string | null;
  roic: number | null;        // Return on Invested Capital (%)
  roe: number | null;         // Return on Equity (%)
  roa: number | null;         // Return on Assets (%)
  profit_margin: number | null;
  operating_margin: number | null;
  beta: number | null;
  market_cap: number | null;
  forward_pe: number | null;
  dividend_yield: number | null;
}

export interface StockOptionsData {
  ticker: string;
  call_open_interest: number | null;
  put_open_interest: number | null;
  total_open_interest: number | null;
  call_volume: number | null;
  put_volume: number | null;
  total_volume: number | null;
  call_put_ratio_oi: number | null;      // Call/Put ratio by open interest
  call_put_ratio_volume: number | null;   // Call/Put ratio by volume
  avg_implied_volatility: number | null;  // Average IV (%)
  iv_percentile: number | null;           // IV percentile (0-100)
  options_sentiment: 'bullish' | 'bearish' | 'neutral' | null;
  has_options: boolean;
}

export interface SectorCorrelation {
  ticker: string;
  sector: string | null;
  industry: string | null;
  sector_etf: string | null;      // ETF used for correlation (e.g., XLK)
  correlation: number | null;      // -1 to 1
  beta_to_sector: number | null;   // Beta relative to sector
}

export interface StockAnalysisData {
  ticker: string;
  fundamentals: StockFundamentals;
  options: StockOptionsData;
  fetched_at: string;
}

/**
 * Get comprehensive analysis data for a stock.
 * Combines fundamentals and options data in one call.
 */
export async function getStockAnalysis(ticker: string): Promise<StockAnalysisData> {
  const response = await fetch(`${API_BASE}/stock/${ticker}/analysis`);
  return handleResponse<StockAnalysisData>(response);
}

/**
 * Get fundamental data for a stock.
 * Includes ROIC, ROE, ROA, sector, industry, margins, beta.
 */
export async function getStockFundamentals(ticker: string): Promise<StockFundamentals> {
  const response = await fetch(`${API_BASE}/stock/${ticker}/fundamentals`);
  return handleResponse<StockFundamentals>(response);
}

/**
 * Get options data for a stock.
 * Includes call/put ratios, open interest, implied volatility.
 */
export async function getStockOptions(ticker: string): Promise<StockOptionsData> {
  const response = await fetch(`${API_BASE}/stock/${ticker}/options`);
  return handleResponse<StockOptionsData>(response);
}

/**
 * Get sector correlation data for a stock.
 * Includes correlation with sector ETF and beta to sector.
 */
export async function getSectorCorrelation(ticker: string, days: number = 60): Promise<SectorCorrelation> {
  const response = await fetch(`${API_BASE}/stock/${ticker}/sector-correlation?days=${days}`);
  return handleResponse<SectorCorrelation>(response);
}
