/**
 * TypeScript types for Walk-Forward Optimization (WFO) Calibration
 * 
 * Isolated from main types/index.ts to keep calibration code separate.
 */

// ============================================================================
// Weight Types
// ============================================================================

/**
 * Indicator weight configuration.
 * Maps indicator names to their weight values (0.0 to 2.5).
 */
export interface WeightMatrix {
  rsi: number;
  macd: number;
  bollinger: number;
  adx: number;
  cmf: number;
  momentum: number;
  volume: number;
  rvol: number;
  sma: number;
  position: number;
  squeeze: number;
  cross?: number;
}

/**
 * Calibrated weight record from database.
 */
export interface CalibrationWeight {
  ticker: string;
  indicator: string;
  action: string;
  horizon: number;
  weight: number;
  sqn_score: number | null;
  stability_passed: boolean;
  calibrated_at: string;
}

// ============================================================================
// Window Types
// ============================================================================

/**
 * Rolling window configuration for WFO.
 */
export interface CalibrationWindow {
  id: number;
  ticker: string;
  horizon: number;
  train_start: string;
  train_end: string;
  test_start: string;
  test_end: string;
  window_days: number;
  weights_json: string;
  train_sqn: number;
  test_sqn: number;
  expectancy: number;
  trades_count: number;
  created_at: string;
}

/**
 * Simulated trade from WFO backtest.
 */
export interface SimulatedTrade {
  id: number;
  window_id: number;
  ticker: string;
  entry_date: string;
  exit_date: string;
  horizon: number;
  direction: 'long' | 'short';
  action: string;
  entry_price: number;
  exit_price: number;
  pnl_pct: number;
  transaction_cost: number;
  market_regime: MarketRegime | null;
}

// ============================================================================
// Progress Types (for SSE streaming)
// ============================================================================

/**
 * Real-time progress update during calibration.
 */
export interface CalibrationProgress {
  ticker: string;
  horizon: number;
  stage: CalibrationStage;
  progress: number;  // 0-100
  current_indicator?: string;
  message?: string;
  weights?: Partial<WeightMatrix>;
  train_sqn?: number;
  test_sqn?: number;
}

export type CalibrationStage = 
  | 'loading'     // Loading price history
  | 'optimizing'  // Running coordinate descent
  | 'testing'     // Evaluating on test window
  | 'saving'      // Persisting to database
  | 'complete'    // Successfully finished
  | 'error';      // Failed with error

// ============================================================================
// Result Types
// ============================================================================

/**
 * Result of optimizing a single indicator.
 */
export interface IndicatorOptResult {
  indicator: string;
  optimal_weight: number;
  sqn_score: number;
  stability_passed: boolean;
  coarse_results: Record<number, number>;  // weight -> SQN
  fine_results: Record<number, number>;    // weight -> SQN
}

/**
 * Full optimization result for a ticker/horizon.
 */
export interface OptimizationResult {
  ticker: string;
  horizon: number;
  weights: WeightMatrix;
  train_sqn: number;
  total_trades: number;
  per_indicator: Record<string, IndicatorOptResult>;
  optimized_at: string;
}

/**
 * Calibration API response for a single horizon.
 */
export interface HorizonResult {
  sqn: number | null;
  trades: number;
  weights: Partial<WeightMatrix> | null;
  error: string | null;
}

/**
 * Full calibration API response.
 */
export interface CalibrationResponse {
  status: 'complete' | 'error';
  ticker: string;
  horizons?: number[];
  results?: Record<number, HorizonResult>;
  error?: string;
  error_code?: 'INSUFFICIENT_VOLATILITY' | 'INSUFFICIENT_DATA';
}

// ============================================================================
// Weight Drift Types
// ============================================================================

/**
 * Weight drift analysis for detecting regime changes.
 */
export interface WeightDrift {
  indicator: string;
  old_weight: number;
  new_weight: number;
  drift: number;  // Absolute change
  stability: 'stable' | 'moderate' | 'noisy';
}

/**
 * Historical weight record for drift visualization.
 */
export interface WeightHistory {
  ticker: string;
  horizon: number;
  date: string;
  weights: Partial<WeightMatrix>;
  sqn: number;
}

// ============================================================================
// Market Regime Types
// ============================================================================

/**
 * 6-state market regime classification.
 */
export type MarketRegime = 
  | 'BULL_QUIET'       // Trending up, low volatility (grind up)
  | 'BULL_VOLATILE'    // Trending up, high volatility (shakeouts)
  | 'BEAR_QUIET'       // Trending down, low volatility (slow bleed)
  | 'BEAR_VOLATILE'    // Trending down, high volatility (crash mode)
  | 'NEUTRAL_CHOP'     // Flat, low volatility (range-bound)
  | 'NEUTRAL_VOLATILE'; // Flat, high volatility (breakout imminent)

/**
 * Regime-specific strategy recommendations.
 */
export const REGIME_STRATEGIES: Record<MarketRegime, string> = {
  BULL_QUIET: 'Buy Calls, Trend Following',
  BULL_VOLATILE: 'Sell Puts (Dip Buying)',
  BEAR_QUIET: 'Slow Bleed - Avoid',
  BEAR_VOLATILE: 'Long Puts (Crash Mode) - Force Cash',
  NEUTRAL_CHOP: 'Mean Reversion, Iron Condors',
  NEUTRAL_VOLATILE: 'Breakout Imminent - Wait'
};

// ============================================================================
// API Request/Response Types
// ============================================================================

/**
 * Request to start calibration.
 */
export interface CalibrationRequest {
  ticker: string;
  horizons?: number[];  // Default: [3, 15]
}

/**
 * Response for weight query.
 */
export interface WeightsResponse {
  ticker: string;
  horizon: number;
  weights: WeightMatrix;
  calibrated_at?: string;
  is_default: boolean;
}

/**
 * Batch weights response.
 */
export interface BatchWeightsResponse {
  horizon: number;
  tickers: Record<string, {
    weights: WeightMatrix;
    is_default: boolean;
    sqn: number | null;
    updated_at: string | null;
  }>;
}

