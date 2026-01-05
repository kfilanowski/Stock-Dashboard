import type { Portfolio, StockData, PortfolioHistoryPoint, StockHistoryResponse } from '../types';

const API_BASE = '/api';

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || 'Request failed');
  }
  return response.json();
}

// Portfolio API
export async function getPortfolio(lite = false): Promise<Portfolio> {
  const url = lite ? `${API_BASE}/portfolio?lite=true` : `${API_BASE}/portfolio`;
  const response = await fetch(url);
  return handleResponse<Portfolio>(response);
}

export async function updatePortfolio(data: { name?: string; total_value?: number }): Promise<Portfolio> {
  const response = await fetch(`${API_BASE}/portfolio`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return handleResponse<Portfolio>(response);
}

// Holdings API
export interface AddHoldingData {
  ticker: string;
  allocation_pct: number;
  investment_date?: string | null;
  investment_price?: number | null;
}

export async function addHolding(data: AddHoldingData): Promise<void> {
  const response = await fetch(`${API_BASE}/holdings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return handleResponse<void>(response);
}

export async function deleteHolding(holdingId: number): Promise<void> {
  const response = await fetch(`${API_BASE}/holdings/${holdingId}`, {
    method: 'DELETE',
  });
  return handleResponse<void>(response);
}

export interface UpdateHoldingData {
  allocation_pct?: number;
  investment_date?: string | null;
  investment_price?: number | null;
}

export async function updateHolding(holdingId: number, data: UpdateHoldingData): Promise<void> {
  const response = await fetch(`${API_BASE}/holdings/${holdingId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return handleResponse<void>(response);
}

// Stock API
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

export async function getStockQuote(ticker: string): Promise<StockQuote> {
  const response = await fetch(`${API_BASE}/stock/${ticker}/quote`);
  return handleResponse<StockQuote>(response);
}

export async function getStockHistory(ticker: string, period: string = '1y'): Promise<StockHistoryResponse> {
  const response = await fetch(`${API_BASE}/stock/${ticker}/history?period=${period}`);
  return handleResponse<StockHistoryResponse>(response);
}

export async function getMultipleStockHistories(
  tickers: string[], 
  period: string = '1y'
): Promise<Record<string, StockHistoryResponse>> {
  const response = await fetch(`${API_BASE}/stocks/history?period=${period}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(tickers),
  });
  return handleResponse<Record<string, StockHistoryResponse>>(response);
}

// Portfolio History
export async function getPortfolioHistory(): Promise<{ history: PortfolioHistoryPoint[] }> {
  const response = await fetch(`${API_BASE}/portfolio/history`);
  return handleResponse(response);
}

export async function createPortfolioSnapshot(): Promise<void> {
  const response = await fetch(`${API_BASE}/portfolio/snapshot`, { method: 'POST' });
  return handleResponse<void>(response);
}

