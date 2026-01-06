export interface Holding {
  id: number;
  portfolio_id: number;
  ticker: string;
  allocation_pct: number;
  added_at: string;
  investment_date?: string | null;
  investment_price?: number | null;
  current_price?: number;
  current_value?: number;
  ytd_return?: number;
  sma_200?: number;
  price_vs_sma?: number;
  gain_loss?: number | null;  // Gain/loss since investment_date in dollars
  gain_loss_pct?: number | null;  // Gain/loss since investment_date in %
}

export interface Portfolio {
  id: number;
  name: string;
  total_value: number;
  created_at: string;
  updated_at: string;
  holdings: Holding[];
  current_total_value?: number;
  total_gain_loss?: number;
  total_gain_loss_pct?: number;
  // User preferences (persisted in database)
  chart_period?: string;
  sort_field?: string;
  sort_direction?: string;
}

export interface StockData {
  ticker: string;
  current_price: number;
  previous_close: number;
  change: number;
  change_pct: number;
  ytd_return: number;
  sma_200: number | null;
  price_vs_sma: number | null;
  high_52w: number | null;
  low_52w: number | null;
  history: HistoryPoint[];
}

export interface HistoryPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface StockHistoryResponse {
  ticker: string;
  history: HistoryPoint[];
  reference_close: number | null;
  is_complete: boolean;
  expected_start: string | null;
  actual_start: string | null;
}

