export interface Holding {
  id: number;
  portfolio_id: number;
  ticker: string;
  shares: number;  // Number of shares owned
  avg_cost?: number | null;  // Average cost per share
  added_at: string;
  is_pinned?: boolean;  // Pin to top of holdings list
  current_price?: number;
  market_value?: number | null;  // shares × current_price
  cost_basis?: number | null;  // shares × avg_cost
  allocation_pct?: number | null;  // Calculated: market_value / total_market_value × 100
  ytd_return?: number;
  sma_200?: number;
  price_vs_sma?: number;
  gain_loss?: number | null;  // market_value - cost_basis
  gain_loss_pct?: number | null;  // (current_price - avg_cost) / avg_cost × 100
  high_52w?: number | null;  // 52-week high price
  low_52w?: number | null;   // 52-week low price
}

export interface Portfolio {
  id: number;
  name: string;
  total_value: number;
  created_at: string;
  updated_at: string;
  holdings: Holding[];
  total_market_value?: number;  // Sum of all holdings' market values
  total_cost_basis?: number | null;  // Sum of all cost basis
  total_gain_loss?: number | null;
  total_gain_loss_pct?: number | null;
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

// ============================================================================
// Stock Scoring Types
// ============================================================================

/**
 * Trading actions that can be scored/recommended.
 * 
 * Options terminology:
 * - openCSP: Open a Cash-Secured Put position (sell a put to collect premium)
 * - openCC: Open a Covered Call position (sell a call to collect premium)
 * - buyCall/buyPut: Buy options (pay premium, long gamma/vega)
 */
export type ActionType = 
  | 'buyShares'
  | 'sellShares'
  | 'openCSP'     // Open Cash-Secured Put (sell put to collect premium)
  | 'openCC'      // Open Covered Call (sell call to collect premium)
  | 'buyCall'
  | 'buyPut';

/**
 * Display names for actions.
 */
export const ACTION_LABELS: Record<ActionType, string> = {
  buyShares: 'Buy Shares',
  sellShares: 'Sell Shares',
  openCSP: 'Open CSP',
  openCC: 'Open CC',
  buyCall: 'Buy Call',
  buyPut: 'Buy Put'
};

/**
 * Technical metrics used for scoring.
 */
export type MetricType =
  | 'rsi'
  | 'macd'
  | 'bollingerBands'
  | 'bollingerSqueeze'  // Volatility contraction/expansion
  | 'vwap'
  | 'momentum'
  | 'volume'
  | 'pricePosition'
  | 'smaAlignment'
  | 'rvol'              // Relative Volume
  | 'adx'               // Average Directional Index (trend strength)
  | 'crossPattern'      // Golden Cross / Death Cross
  | 'cmf'               // Chaikin Money Flow (accumulation/distribution)
  | 'divergence'        // Price-Volume Divergence
  | 'alpha'             // Relative strength vs market (SPY)
  | 'roic'              // Return on Invested Capital
  | 'callPutRatio'      // Options Call/Put ratio
  | 'ivPercentile'      // Implied Volatility percentile
  | 'sectorBeta'        // Beta to sector
  | 'earningsProximity'; // Days until next earnings

/**
 * A signal from a single metric for a specific action.
 * Signal ranges from -1 (strongly against) to +1 (strongly for).
 */
export interface MetricSignal {
  metric: MetricType;
  metricLabel: string;
  rawValue: string;           // Human-readable value (e.g., "RSI: 25")
  signal: number;             // -1 to +1
  weight: number;             // Weight for this metric (default 1.0)
  contribution: number;       // signal * weight
  reasoning: string;          // Why this signal was given
}

/**
 * Score for a single action, with breakdown by metric.
 */
export interface ActionScore {
  action: ActionType;
  label: string;
  totalScore: number;         // Sum of all weighted signals, normalized 0-100
  rawScore: number;           // Sum before normalization
  signals: MetricSignal[];    // Individual metric contributions
  confidence: 'high' | 'medium' | 'low';  // Based on data availability
}

/**
 * Complete stock analysis with all action scores.
 */
export interface StockAnalysis {
  ticker: string;
  analyzedAt: Date;
  scores: ActionScore[];
  bestAction: ActionScore;
  hasOptions: boolean;          // Whether options are available for this stock
  dataQuality: {
    availableMetrics: number;
    totalMetrics: number;
    missingMetrics: MetricType[];
    historyDays: number;        // How many days of history were used
    hasSMA200: boolean;         // Whether SMA200 could be calculated
  };
}

// ============================================================================
// Option Holdings Types
// ============================================================================

export type OptionType = 'call' | 'put';
export type PositionType = 'long' | 'short';

/**
 * Greeks for an option position.
 */
export interface OptionGreeks {
  delta: number | null;   // Rate of change vs underlying price
  gamma: number | null;   // Rate of change of delta
  theta: number | null;   // Time decay (per day)
  vega: number | null;    // Sensitivity to IV changes
  rho: number | null;     // Sensitivity to interest rate
}

/**
 * Calculated analytics for an option position.
 */
export interface OptionAnalytics {
  breakeven_price: number | null;    // Price at which position breaks even
  max_profit: number | null;         // Maximum possible profit (null if unlimited)
  max_loss: number | null;           // Maximum possible loss (null if unlimited)
  profit_probability: number | null; // Estimated probability of profit (0-100)
  days_to_expiration: number;
  is_itm: boolean;                   // In the money
  is_expired: boolean;
  intrinsic_value: number | null;    // Current intrinsic value per contract
  time_value: number | null;         // Current time/extrinsic value per contract
}

/**
 * Base option holding interface (for creation).
 */
export interface OptionHoldingCreate {
  underlying_ticker: string;
  option_type: OptionType;
  position_type: PositionType;
  strike_price: number;
  expiration_date: string;  // ISO date string YYYY-MM-DD
  contracts: number;
  premium_per_contract?: number | null;
  notes?: string | null;
}

/**
 * Option holding update interface.
 */
export interface OptionHoldingUpdate {
  contracts?: number;
  premium_per_contract?: number | null;
  notes?: string | null;
}

/**
 * Base option holding from database.
 */
export interface OptionHolding {
  id: number;
  portfolio_id: number;
  underlying_ticker: string;
  option_type: OptionType;
  position_type: PositionType;
  strike_price: number;
  expiration_date: string;
  contracts: number;
  premium_per_contract: number | null;
  opened_at: string;
  notes: string | null;
}

/**
 * Option holding with live market data and analytics.
 */
export interface OptionHoldingWithData extends OptionHolding {
  // Current market data
  underlying_price: number | null;
  current_price: number | null;      // Current option price per share
  bid: number | null;
  ask: number | null;
  implied_volatility: number | null; // IV as percentage
  open_interest: number | null;
  volume: number | null;
  
  // Position values
  position_value: number | null;     // contracts * 100 * current_price
  cost_basis: number | null;         // contracts * 100 * premium_per_contract
  gain_loss: number | null;          // P/L in dollars
  gain_loss_pct: number | null;      // P/L percentage
  
  // Greeks
  greeks: OptionGreeks | null;
  
  // Analytics
  analytics: OptionAnalytics | null;
}

/**
 * Display labels for option types.
 */
export const OPTION_TYPE_LABELS: Record<OptionType, string> = {
  call: 'Call',
  put: 'Put'
};

/**
 * Display labels for position types.
 */
export const POSITION_TYPE_LABELS: Record<PositionType, string> = {
  long: 'Long',
  short: 'Short'
};

