/**
 * Option Position Scoring Service - Strategy-Aware Risk Management
 *
 * KEY PRINCIPLES:
 * - Short Options (Income/Wheel): Success = time decay + volatility crush. 
 *   Close early to free capital (Velocity of Money). Fear Gamma risk.
 * - Long Options (Speculation): Success = momentum + convexity.
 *   Let winners run to capture gamma expansion. Fear Theta decay.
 *
 * CRITICAL THRESHOLDS:
 * - 21 DTE: "Gamma Cliff" for shorts - risk accelerates faster than reward
 * - 50% profit on shorts: Optimal exit per Velocity of Money doctrine
 * - IV Rank 60+: Sell premium zone / IV Crush incoming
 * - IV Rank <30: Buy premium zone / cheap options
 *
 * OVERRIDE LOGIC ("Kill Switches"):
 * - DTE <= 1: Force CLOSE (pin risk / assignment)
 * - Short at 90%+ profit: Force CLOSE (no upside left)
 * - Long at 200%+ profit: Signal CLOSE (take the win)
 * - Spread > 15%: Warn about exit difficulty
 */

import type { OptionHoldingWithData } from '../types';

// ============================================================================
// Types
// ============================================================================

export type OptionAction = 'OPEN_MORE' | 'HOLD' | 'CLOSE';
export type SignalType = 'bullish' | 'bearish' | 'neutral';
export type FactorCategory = 'time' | 'profit' | 'greeks' | 'market' | 'structure';

export interface FactorAnalysis {
  name: string;
  value: string;
  signal: SignalType;
  weight: number;
  reasoning: string;
  category: FactorCategory;
}

export interface OptionRecommendation {
  action: OptionAction;
  confidence: number;
  reasons: string[];
  factors: FactorAnalysis[];
  summary: string;
  riskLevel: 'low' | 'medium' | 'high';
  isOverride: boolean;  // True if a "kill switch" forced the decision
}

export interface StockAnalysisData {
  iv_percentile?: number | null;
  options_sentiment?: string | null;
  next_earnings_date?: string | Date | null;  // For binary event risk
  rsi?: number | null;                         // For mean reversion timing
}

// ============================================================================
// Strategy-Aware Thresholds
// ============================================================================

const THRESHOLDS = {
  // DTE Zones - Different for Long vs Short
  SHORT_GAMMA_CLIFF: 21,       // Shorts: Gamma risk > Theta reward after this
  SHORT_THETA_SWEET: 45,       // Shorts: Optimal theta decay curve
  LONG_THETA_DANGER: 14,       // Longs: Theta acceleration hurts
  LONG_THETA_CRITICAL: 7,      // Longs: Exponential decay zone
  EXPIRY_PIN_RISK: 3,          // Both: Assignment/pin risk zone
  EXPIRY_CRITICAL: 1,          // Both: Close immediately

  // Profit Management - Strategy Specific
  SHORT_TAKE_PROFIT_EARLY: 50, // "Velocity of Money" - free up capital
  SHORT_TAKE_PROFIT_FULL: 80,  // Squeeze last drops if DTE allows
  SHORT_MAX_PROFIT: 90,        // Kill switch: no upside left
  LONG_TRIM_PROFIT: 100,       // Consider trimming longs
  LONG_TAKE_PROFIT: 200,       // Take the win on longs
  
  // Stop Loss
  STOP_LOSS_STANDARD: -50,     // Standard mental stop
  STOP_LOSS_WIDE: -100,        // 2x loss for short premium
  
  // IV Rank (Mean Reversion Logic)
  IV_SELL_ZONE: 60,            // High IV: Sell premium, expect crush
  IV_NEUTRAL_HIGH: 50,         // Neutral zone
  IV_NEUTRAL_LOW: 35,          // Neutral zone  
  IV_BUY_ZONE: 25,             // Low IV: Buy premium, cheap options

  // Greeks
  DELTA_ASSIGNMENT_RISK: 0.70, // High probability ITM for shorts
  DELTA_LOTTERY_TICKET: 0.15,  // Low probability for longs - needs big move
  GAMMA_DANGEROUS: 0.08,       // High gamma = rapid delta swings
  THETA_EFFICIENT: 0.03,       // 3%+ daily decay is efficient for shorts

  // Liquidity
  SPREAD_WIDE: 0.10,           // >10% spread is problematic
  SPREAD_ILLIQUID: 0.20,       // >20% = very hard to exit
  
  // Open Interest / Volume (Ghost Liquidity)
  OI_ZOMBIE: 50,               // <50 OI = potentially trapped
  OI_MINIMUM: 100,             // Minimum safe OI
  OI_HEALTHY: 500,             // Good liquidity
  VOLUME_HIGH_RATIO: 1.0,      // Volume > OI = easy execution
  
  // RSI Thresholds (Mean Reversion)
  RSI_OVERSOLD: 30,            // Entry zone for bullish positions
  RSI_OVERBOUGHT: 70,          // Entry zone for bearish positions
  RSI_EXTREME_LOW: 20,         // Extreme oversold
  RSI_EXTREME_HIGH: 80,        // Extreme overbought
  
  // Earnings Risk
  EARNINGS_DANGER_ZONE: 7,     // Days before earnings = high risk
  EARNINGS_WARNING: 14,        // Days before earnings = elevated risk
};

// ============================================================================
// Display Constants
// ============================================================================

export const ACTION_LABELS: Record<OptionAction, string> = {
  OPEN_MORE: 'Open More',
  HOLD: 'Hold',
  CLOSE: 'Close',
};

export const ACTION_COLORS: Record<OptionAction, { bg: string; text: string; border: string }> = {
  CLOSE: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/30' },
  HOLD: { bg: 'bg-white/10', text: 'text-white/70', border: 'border-white/20' },
  OPEN_MORE: { bg: 'bg-green-500/20', text: 'text-green-400', border: 'border-green-500/30' },
};

export const CATEGORY_LABELS: Record<FactorCategory, string> = {
  time: 'Time & Decay',
  profit: 'Profit & Loss',
  greeks: 'Greeks & Risk',
  market: 'Market Conditions',
  structure: 'Position Structure',
};

// ============================================================================
// Main Analysis Function
// ============================================================================

export function analyzeOptionPosition(
  option: OptionHoldingWithData,
  stockAnalysis?: StockAnalysisData
): OptionRecommendation {
  const factors: FactorAnalysis[] = [];
  const reasons: string[] = [];

  // Extract core position data
  const isLong = option.position_type === 'long';
  const isCall = option.option_type === 'call';
  const dte = option.analytics?.days_to_expiration ?? 0;
  const gainLossPct = option.gain_loss_pct ?? 0;
  const profitProb = option.analytics?.profit_probability ?? null;
  const isItm = option.analytics?.is_itm ?? false;
  const ivRank = stockAnalysis?.iv_percentile ?? null;
  const sentiment = stockAnalysis?.options_sentiment ?? null;
  const underlyingPrice = option.underlying_price ?? 0;
  const strikePrice = option.strike_price;

  // Determine strategy type for clearer logic
  const isBullishPosition = (isCall && isLong) || (!isCall && !isLong);
  
  // Scoring accumulators
  let closeScore = 0;
  let holdScore = 0;
  let openMoreScore = 0;
  let riskScore = 0;

  // ============================================================================
  // KILL SWITCHES - Override all other logic
  // ============================================================================

  // Check for immediate override conditions
  const override = checkOverrides(option, isLong, dte, gainLossPct);
  if (override) {
    return override;
  }

  // ============================================================================
  // FACTOR 1: Time Risk (Strategy-Specific)
  // ============================================================================
  
  const timeFactor = analyzeTimeRisk(dte, isLong, isItm);
  factors.push(timeFactor);

  if (isLong) {
    // LONG: Time is the enemy - theta works against us
    if (timeFactor.signal === 'bearish') {
      closeScore += timeFactor.weight * 35;
      riskScore += 25;
      reasons.push(timeFactor.reasoning);
    } else if (timeFactor.signal === 'bullish') {
      holdScore += timeFactor.weight * 15;
      openMoreScore += timeFactor.weight * 10;
    } else {
      holdScore += timeFactor.weight * 10;
    }
  } else {
    // SHORT: Time is our friend, BUT gamma risk increases near expiry
    if (dte <= THRESHOLDS.SHORT_GAMMA_CLIFF) {
      // Past the Gamma Cliff - risk > reward
      closeScore += timeFactor.weight * 30;
      riskScore += 20;
      if (!reasons.some(r => r.includes('Gamma'))) {
        reasons.push(`Past Gamma Cliff (${THRESHOLDS.SHORT_GAMMA_CLIFF}d) - risk outweighs theta`);
      }
    } else if (timeFactor.signal === 'bullish') {
      // In the theta sweet spot
      holdScore += timeFactor.weight * 20;
      openMoreScore += timeFactor.weight * 10;
    } else {
      holdScore += timeFactor.weight * 10;
    }
  }

  // ============================================================================
  // FACTOR 2: Profit Management (Velocity of Money for Shorts)
  // ============================================================================

  const profitFactor = analyzeProfitManagement(gainLossPct, isLong, dte);
  factors.push(profitFactor);

  if (isLong) {
    // LONG: Let winners run (convexity), cut losers
    if (gainLossPct >= THRESHOLDS.LONG_TAKE_PROFIT) {
      closeScore += profitFactor.weight * 30;
      reasons.push(`Exceptional ${gainLossPct.toFixed(0)}% gain - lock in profits`);
    } else if (gainLossPct >= THRESHOLDS.LONG_TRIM_PROFIT) {
      holdScore += profitFactor.weight * 15;
      closeScore += profitFactor.weight * 10; // Consider trimming
    } else if (gainLossPct <= THRESHOLDS.STOP_LOSS_STANDARD) {
      closeScore += profitFactor.weight * 25;
      riskScore += 30;
      reasons.push(`Down ${Math.abs(gainLossPct).toFixed(0)}% - stop loss discipline`);
    } else if (gainLossPct > 0) {
      holdScore += profitFactor.weight * 15;
    } else {
      holdScore += profitFactor.weight * 5;
    }
  } else {
    // SHORT: Velocity of Money - take profits early to free capital
    if (gainLossPct >= THRESHOLDS.SHORT_TAKE_PROFIT_EARLY) {
      // 50%+ profit: Optimal exit point
      closeScore += profitFactor.weight * 40;
      reasons.push(`${gainLossPct.toFixed(0)}% profit - free up capital for new trades`);
    } else if (gainLossPct > 0) {
      // Profitable but not at target - hold for more decay
      holdScore += profitFactor.weight * 20;
    } else if (gainLossPct <= THRESHOLDS.STOP_LOSS_WIDE) {
      // 2x loss on short premium
      closeScore += profitFactor.weight * 30;
      riskScore += 35;
      reasons.push(`Position breached 2x loss limit`);
    } else if (gainLossPct < -25) {
      holdScore += profitFactor.weight * 5;
      riskScore += 15;
    } else {
      holdScore += profitFactor.weight * 10;
    }
  }

  // ============================================================================
  // FACTOR 3: Profit Probability
  // ============================================================================

  if (profitProb !== null) {
    const probFactor = analyzeProbability(profitProb, gainLossPct, isLong);
    factors.push(probFactor);

    if (probFactor.signal === 'bearish') {
      closeScore += probFactor.weight * 20;
      riskScore += 15;
      if (profitProb < 20 && gainLossPct < 0) {
        reasons.push(`Only ${profitProb.toFixed(0)}% chance of recovery`);
      }
    } else if (probFactor.signal === 'bullish') {
      holdScore += probFactor.weight * 15;
      openMoreScore += probFactor.weight * 8;
    } else {
      holdScore += probFactor.weight * 8;
    }
  }

  // ============================================================================
  // FACTOR 4: IV Environment (Mean Reversion Logic)
  // ============================================================================

  if (ivRank !== null) {
    const ivFactor = analyzeIVEnvironment(ivRank, isLong);
    factors.push(ivFactor);

    if (isLong) {
      // LONG: Low IV is good (cheap), High IV is risky (crush incoming)
      if (ivRank >= THRESHOLDS.IV_SELL_ZONE) {
        closeScore += ivFactor.weight * 20;
        riskScore += 15;
        reasons.push(`IV Rank ${ivRank} - crush risk for long positions`);
      } else if (ivRank <= THRESHOLDS.IV_BUY_ZONE) {
        openMoreScore += ivFactor.weight * 20;
        holdScore += ivFactor.weight * 10;
      } else {
        holdScore += ivFactor.weight * 8;
      }
    } else {
      // SHORT: High IV is good (premium + crush), Low IV reduces edge
      if (ivRank >= THRESHOLDS.IV_SELL_ZONE) {
        holdScore += ivFactor.weight * 20; // Wait for crush
        openMoreScore += ivFactor.weight * 15;
      } else if (ivRank <= THRESHOLDS.IV_BUY_ZONE) {
        closeScore += ivFactor.weight * 15;
        reasons.push(`IV Rank ${ivRank} - low premium environment`);
      } else {
        holdScore += ivFactor.weight * 10;
      }
    }
  }

  // ============================================================================
  // FACTOR 5: Market Sentiment Alignment
  // ============================================================================

  if (sentiment) {
    const sentimentFactor = analyzeMarketAlignment(sentiment, isBullishPosition);
    factors.push(sentimentFactor);

    if (sentimentFactor.signal === 'bullish') {
      holdScore += sentimentFactor.weight * 15;
      openMoreScore += sentimentFactor.weight * 12;
    } else if (sentimentFactor.signal === 'bearish') {
      closeScore += sentimentFactor.weight * 15;
      riskScore += 10;
      reasons.push('Fighting market flow');
    } else {
      holdScore += sentimentFactor.weight * 5;
    }
  }

  // ============================================================================
  // FACTOR 6: Delta / Assignment Risk (Critical for Shorts)
  // ============================================================================

  if (option.greeks && option.greeks.delta !== null) {
    const deltaFactor = analyzeDeltaRisk(option.greeks.delta, isLong, isCall);
    factors.push(deltaFactor);

    if (!isLong && deltaFactor.signal === 'bearish') {
      // High delta on short = assignment risk
      closeScore += deltaFactor.weight * 25;
      riskScore += 20;
      reasons.push('High assignment probability');
    } else if (isLong && deltaFactor.signal === 'bearish') {
      // Low delta on long = lottery ticket
      closeScore += deltaFactor.weight * 15;
      riskScore += 10;
    } else if (deltaFactor.signal === 'bullish') {
      holdScore += deltaFactor.weight * 10;
    }
  }

  // ============================================================================
  // FACTOR 7: Gamma Risk (Especially for Shorts near Expiry)
  // ============================================================================

  if (option.greeks && option.greeks.gamma !== null) {
    const gammaFactor = analyzeGammaRisk(option.greeks.gamma, dte, isLong);
    factors.push(gammaFactor);

    if (gammaFactor.signal === 'bearish') {
      closeScore += gammaFactor.weight * 20;
      riskScore += 25;
      if (!isLong) {
        reasons.push('High gamma exposure - price swings hurt');
      }
    } else {
      holdScore += gammaFactor.weight * 5;
    }
  }

  // ============================================================================
  // FACTOR 8: Theta Efficiency (Shorts)
  // ============================================================================

  if (!isLong && option.greeks && option.greeks.theta !== null && option.current_price) {
    const thetaFactor = analyzeThetaEfficiency(option.greeks.theta, option.current_price, dte);
    factors.push(thetaFactor);

    if (thetaFactor.signal === 'bullish') {
      holdScore += thetaFactor.weight * 15;
      openMoreScore += thetaFactor.weight * 8;
    } else if (thetaFactor.signal === 'bearish') {
      closeScore += thetaFactor.weight * 10;
    }
  }

  // ============================================================================
  // FACTOR 9: Liquidity Health (Spread + Volume/OI)
  // ============================================================================

  if (option.bid !== null && option.ask !== null) {
    const liquidityFactor = analyzeLiquidity(
      option.bid, 
      option.ask, 
      option.volume ?? 0, 
      option.open_interest ?? 0
    );
    factors.push(liquidityFactor);

    if (liquidityFactor.signal === 'bearish') {
      riskScore += liquidityFactor.weight * 15;
      if (closeScore > holdScore) {
        reasons.push(liquidityFactor.reasoning);
      }
    }
  }

  // ============================================================================
  // FACTOR 10: Moneyness / Position Structure
  // ============================================================================

  if (underlyingPrice > 0) {
    const moneynessFactor = analyzeMoneyness(underlyingPrice, strikePrice, isCall, isLong, dte);
    factors.push(moneynessFactor);

    if (moneynessFactor.signal === 'bearish') {
      closeScore += moneynessFactor.weight * 15;
      riskScore += 15;
    } else if (moneynessFactor.signal === 'bullish') {
      holdScore += moneynessFactor.weight * 10;
    }
  }

  // ============================================================================
  // FACTOR 11: Earnings Risk (Binary Event)
  // ============================================================================

  if (stockAnalysis?.next_earnings_date) {
    const earningsFactor = analyzeEarningsRisk(dte, stockAnalysis.next_earnings_date, isLong);
    factors.push(earningsFactor);

    if (earningsFactor.signal === 'bearish') {
      closeScore += earningsFactor.weight * 25;
      riskScore += 30;
      reasons.push(earningsFactor.reasoning);
    } else if (earningsFactor.signal === 'bullish') {
      holdScore += earningsFactor.weight * 10;
    }
  }

  // ============================================================================
  // FACTOR 12: RSI Timing (Mean Reversion)
  // ============================================================================

  if (stockAnalysis?.rsi !== undefined && stockAnalysis.rsi !== null) {
    const rsiFactor = analyzeRSI(stockAnalysis.rsi, isLong, isCall);
    factors.push(rsiFactor);

    if (rsiFactor.signal === 'bearish') {
      closeScore += rsiFactor.weight * 15;
      riskScore += 10;
      if (rsiFactor.weight >= 1.5) {
        reasons.push(rsiFactor.reasoning);
      }
    } else if (rsiFactor.signal === 'bullish') {
      holdScore += rsiFactor.weight * 15;
      openMoreScore += rsiFactor.weight * 10;
    }
  }

  // ============================================================================
  // FINAL DECISION
  // ============================================================================

  const totalScore = closeScore + holdScore + openMoreScore;
  let action: OptionAction;
  let confidence: number;

  if (totalScore === 0) {
    action = 'HOLD';
    confidence = 30;
  } else {
    const closeNorm = closeScore / totalScore;
    const holdNorm = holdScore / totalScore;
    const openNorm = openMoreScore / totalScore;

    if (closeNorm > holdNorm && closeNorm > openNorm) {
      action = 'CLOSE';
      confidence = Math.min(90, Math.round(closeNorm * 100) + 15);
    } else if (openNorm > holdNorm && openNorm > closeNorm) {
      action = 'OPEN_MORE';
      confidence = Math.min(85, Math.round(openNorm * 100) + 10);
    } else {
      action = 'HOLD';
      confidence = Math.min(80, Math.round(holdNorm * 100) + 10);
    }
  }

  // Calculate risk level
  const maxPossibleRisk = factors.length * 25;
  const riskPct = maxPossibleRisk > 0 ? riskScore / maxPossibleRisk : 0;
  const riskLevel: 'low' | 'medium' | 'high' = 
    riskPct > 0.5 ? 'high' : riskPct > 0.25 ? 'medium' : 'low';

  // Ensure at least one reason
  if (reasons.length === 0) {
    reasons.push(getDefaultReason(action, option));
  }

  // Sort factors by category
  factors.sort((a, b) => {
    const order: FactorCategory[] = ['time', 'profit', 'greeks', 'market', 'structure'];
    return order.indexOf(a.category) - order.indexOf(b.category);
  });

  return {
    action,
    confidence,
    reasons: reasons.slice(0, 4),
    factors,
    summary: generateSummary(action, option, reasons[0]),
    riskLevel,
    isOverride: false,
  };
}

// ============================================================================
// Kill Switch / Override Logic
// ============================================================================

function checkOverrides(
  option: OptionHoldingWithData,
  isLong: boolean,
  dte: number,
  gainLossPct: number
): OptionRecommendation | null {
  const factors: FactorAnalysis[] = [];
  
  // OVERRIDE 1: Expiration Imminent (1 day or less)
  if (dte <= THRESHOLDS.EXPIRY_CRITICAL) {
    factors.push({
      name: 'Expiration Override',
      value: `${dte} day(s)`,
      signal: 'bearish',
      weight: 5,
      reasoning: 'Expiration imminent - close to avoid pin risk and assignment',
      category: 'time',
    });
    
    return {
      action: 'CLOSE',
      confidence: 95,
      reasons: ['Expiration imminent - close to avoid pin risk/assignment'],
      factors,
      summary: generateSummary('CLOSE', option, 'Expiration imminent'),
      riskLevel: 'high',
      isOverride: true,
    };
  }

  // OVERRIDE 2: Short at 90%+ profit (no upside left)
  if (!isLong && gainLossPct >= THRESHOLDS.SHORT_MAX_PROFIT) {
    factors.push({
      name: 'Max Profit Override',
      value: `+${gainLossPct.toFixed(0)}%`,
      signal: 'bullish',
      weight: 5,
      reasoning: 'Max profit effectively realized - no meaningful upside remaining',
      category: 'profit',
    });
    
    return {
      action: 'CLOSE',
      confidence: 92,
      reasons: ['Max profit realized - no upside left, only risk'],
      factors,
      summary: generateSummary('CLOSE', option, 'Max profit realized'),
      riskLevel: 'low',
      isOverride: true,
    };
  }

  // OVERRIDE 3: Long at 200%+ profit (exceptional win)
  if (isLong && gainLossPct >= THRESHOLDS.LONG_TAKE_PROFIT) {
    factors.push({
      name: 'Exceptional Gain Override',
      value: `+${gainLossPct.toFixed(0)}%`,
      signal: 'bullish',
      weight: 5,
      reasoning: 'Exceptional gain - strongly consider locking in profits',
      category: 'profit',
    });
    
    return {
      action: 'CLOSE',
      confidence: 85,
      reasons: [`Exceptional ${gainLossPct.toFixed(0)}% gain - lock in the win`],
      factors,
      summary: generateSummary('CLOSE', option, 'Exceptional gain'),
      riskLevel: 'low',
      isOverride: true,
    };
  }

  // OVERRIDE 4: Stop loss hit (any position down 50%+)
  if (gainLossPct <= THRESHOLDS.STOP_LOSS_STANDARD) {
    factors.push({
      name: 'Stop Loss Override',
      value: `${gainLossPct.toFixed(0)}%`,
      signal: 'bearish',
      weight: 5,
      reasoning: 'Position breached 50% loss threshold - discipline requires exit',
      category: 'profit',
    });
    
    return {
      action: 'CLOSE',
      confidence: 88,
      reasons: [`Down ${Math.abs(gainLossPct).toFixed(0)}% - stop loss discipline`],
      factors,
      summary: generateSummary('CLOSE', option, 'Stop loss hit'),
      riskLevel: 'high',
      isOverride: true,
    };
  }

  // OVERRIDE 5: Short position with option nearly worthless (capture complete)
  if (!isLong && option.current_price !== null && option.current_price <= 0.05) {
    factors.push({
      name: 'Dead Capital Override',
      value: `$${option.current_price.toFixed(2)}`,
      signal: 'bullish',
      weight: 5,
      reasoning: 'Option nearly worthless - no reason to risk $100 for $5',
      category: 'profit',
    });
    
    return {
      action: 'CLOSE',
      confidence: 90,
      reasons: ['Option worth pennies - close to eliminate risk'],
      factors,
      summary: generateSummary('CLOSE', option, 'Capture complete'),
      riskLevel: 'low',
      isOverride: true,
    };
  }

  return null;
}

// ============================================================================
// Factor Analysis Functions
// ============================================================================

function analyzeTimeRisk(dte: number, isLong: boolean, isItm: boolean): FactorAnalysis {
  let signal: SignalType = 'neutral';
  let reasoning: string;
  let weight = 1.0;

  if (isLong) {
    // LONG: Theta is the enemy
    if (dte <= THRESHOLDS.LONG_THETA_CRITICAL) {
      signal = 'bearish';
      weight = 1.5;
      reasoning = isItm 
        ? `${dte}d left - exercise/close decision needed`
        : `${dte}d left - theta burn is exponential, OTM is critical`;
    } else if (dte <= THRESHOLDS.LONG_THETA_DANGER) {
      signal = 'bearish';
      weight = 1.2;
      reasoning = `${dte}d left - theta acceleration hurting position`;
    } else if (dte >= 45) {
      signal = 'bullish';
      reasoning = `${dte}d remaining - safe time horizon`;
    } else {
      signal = 'neutral';
      reasoning = `${dte}d remaining - manageable decay`;
    }
  } else {
    // SHORT: Theta is our friend until Gamma Cliff
    if (dte <= THRESHOLDS.EXPIRY_PIN_RISK) {
      signal = 'bearish';
      weight = 1.4;
      reasoning = `${dte}d left - pin risk and gamma explosion zone`;
    } else if (dte <= THRESHOLDS.SHORT_GAMMA_CLIFF) {
      signal = 'bearish';
      weight = 1.2;
      reasoning = `${dte}d left - past Gamma Cliff, risk > reward`;
    } else if (dte >= 30 && dte <= THRESHOLDS.SHORT_THETA_SWEET) {
      signal = 'bullish';
      weight = 1.1;
      reasoning = `${dte}d - ideal theta decay curve zone`;
    } else {
      signal = 'neutral';
      reasoning = `${dte}d remaining`;
    }
  }

  return {
    name: 'Time Risk',
    value: `${dte} days`,
    signal,
    weight,
    reasoning,
    category: 'time',
  };
}

function analyzeProfitManagement(gainLossPct: number, isLong: boolean, dte: number): FactorAnalysis {
  let signal: SignalType = 'neutral';
  let reasoning: string;
  let weight = 1.0;

  if (!isLong) {
    // SHORT: Capped upside - take profits systematically (Velocity of Money)
    if (gainLossPct >= THRESHOLDS.SHORT_TAKE_PROFIT_FULL) {
      signal = 'bullish';
      weight = 1.4;
      reasoning = `${gainLossPct.toFixed(0)}% profit - optimal to close and redeploy`;
    } else if (gainLossPct >= THRESHOLDS.SHORT_TAKE_PROFIT_EARLY) {
      signal = 'bullish';
      weight = 1.2;
      reasoning = `${gainLossPct.toFixed(0)}% profit - 50% rule: free up capital`;
    } else if (gainLossPct <= THRESHOLDS.STOP_LOSS_WIDE) {
      signal = 'bearish';
      weight = 1.3;
      reasoning = `Down ${Math.abs(gainLossPct).toFixed(0)}% - 2x loss limit breached`;
    } else if (gainLossPct < -25) {
      signal = 'bearish';
      reasoning = `Down ${Math.abs(gainLossPct).toFixed(0)}% - monitor closely`;
    } else if (gainLossPct > 0) {
      signal = 'neutral';
      reasoning = `Up ${gainLossPct.toFixed(0)}% - holding for more decay`;
    } else {
      signal = 'neutral';
      reasoning = `${gainLossPct >= 0 ? 'Up' : 'Down'} ${Math.abs(gainLossPct).toFixed(0)}%`;
    }
  } else {
    // LONG: Uncapped upside - let winners run
    if (gainLossPct >= THRESHOLDS.LONG_TRIM_PROFIT) {
      signal = 'bullish';
      weight = 1.3;
      reasoning = `Up ${gainLossPct.toFixed(0)}% - consider trimming or rolling`;
    } else if (gainLossPct > 30 && dte < 14) {
      signal = 'bullish';
      weight = 1.2;
      reasoning = `Profitable near expiry - lock in gains`;
    } else if (gainLossPct <= THRESHOLDS.STOP_LOSS_STANDARD) {
      signal = 'bearish';
      weight = 1.3;
      reasoning = `Down ${Math.abs(gainLossPct).toFixed(0)}% - stop loss discipline`;
    } else if (gainLossPct < -20) {
      signal = 'bearish';
      reasoning = `Down ${Math.abs(gainLossPct).toFixed(0)}%`;
    } else {
      signal = 'neutral';
      reasoning = `${gainLossPct >= 0 ? '+' : ''}${gainLossPct.toFixed(0)}%`;
    }
  }

  return {
    name: 'P/L Management',
    value: `${gainLossPct >= 0 ? '+' : ''}${gainLossPct.toFixed(1)}%`,
    signal,
    weight,
    reasoning,
    category: 'profit',
  };
}

function analyzeProbability(profitProb: number, gainLossPct: number, _isLong: boolean): FactorAnalysis {
  let signal: SignalType = 'neutral';
  let reasoning: string;
  let weight = 0.9;

  if (profitProb <= 15) {
    signal = 'bearish';
    weight = 1.1;
    reasoning = `Only ${profitProb.toFixed(0)}% probability - low odds of recovery`;
  } else if (profitProb <= 30 && gainLossPct < 0) {
    signal = 'bearish';
    reasoning = `${profitProb.toFixed(0)}% probability with current loss - risky`;
  } else if (profitProb >= 80) {
    signal = 'bullish';
    reasoning = `${profitProb.toFixed(0)}% probability - strong position`;
  } else if (profitProb >= 60) {
    signal = 'bullish';
    reasoning = `${profitProb.toFixed(0)}% probability favors profit`;
  } else {
    signal = 'neutral';
    reasoning = `${profitProb.toFixed(0)}% profit probability`;
  }

  return {
    name: 'Win Probability',
    value: `${profitProb.toFixed(0)}%`,
    signal,
    weight,
    reasoning,
    category: 'profit',
  };
}

function analyzeIVEnvironment(ivRank: number, isLong: boolean): FactorAnalysis {
  let signal: SignalType = 'neutral';
  let reasoning: string;
  let weight = 1.0;

  // Mean Reversion Logic: High IV tends to drop, Low IV tends to rise

  if (ivRank >= 80) {
    signal = isLong ? 'bearish' : 'bullish';
    weight = 1.2;
    reasoning = isLong
      ? `IV Rank ${ivRank} - very expensive, high crush risk`
      : `IV Rank ${ivRank} - excellent premium, IV crush coming`;
  } else if (ivRank >= THRESHOLDS.IV_SELL_ZONE) {
    signal = isLong ? 'bearish' : 'bullish';
    reasoning = isLong
      ? `IV Rank ${ivRank} - elevated, crush risk`
      : `IV Rank ${ivRank} - good premium environment`;
  } else if (ivRank <= 15) {
    signal = isLong ? 'bullish' : 'bearish';
    weight = 1.1;
    reasoning = isLong
      ? `IV Rank ${ivRank} - very cheap entry`
      : `IV Rank ${ivRank} - minimal premium, vega expansion risk`;
  } else if (ivRank <= THRESHOLDS.IV_BUY_ZONE) {
    signal = isLong ? 'bullish' : 'bearish';
    reasoning = isLong
      ? `IV Rank ${ivRank} - cheap options`
      : `IV Rank ${ivRank} - low premium environment`;
  } else {
    signal = 'neutral';
    reasoning = `IV Rank ${ivRank} - neutral zone`;
  }

  return {
    name: 'IV Rank',
    value: `${ivRank}`,
    signal,
    weight,
    reasoning,
    category: 'market',
  };
}

function analyzeMarketAlignment(sentiment: string, isBullishPosition: boolean): FactorAnalysis {
  let signal: SignalType = 'neutral';
  let reasoning: string;
  const weight = 0.9;

  if (sentiment === 'bullish') {
    if (isBullishPosition) {
      signal = 'bullish';
      reasoning = 'Market flow confirms bullish position';
    } else {
      signal = 'bearish';
      reasoning = 'Bullish flow conflicts with bearish position';
    }
  } else if (sentiment === 'bearish') {
    if (!isBullishPosition) {
      signal = 'bullish';
      reasoning = 'Market flow confirms bearish position';
    } else {
      signal = 'bearish';
      reasoning = 'Bearish flow conflicts with bullish position';
    }
  } else {
    signal = 'neutral';
    reasoning = 'Neutral market sentiment';
  }

  return {
    name: 'Market Flow',
    value: sentiment.charAt(0).toUpperCase() + sentiment.slice(1),
    signal,
    weight,
    reasoning,
    category: 'market',
  };
}

function analyzeDeltaRisk(delta: number, isLong: boolean, _isCall: boolean): FactorAnalysis {
  const absDelta = Math.abs(delta);
  let signal: SignalType = 'neutral';
  let reasoning: string;
  let weight = 0.8;

  if (!isLong) {
    // SHORT: High delta = assignment risk
    if (absDelta >= THRESHOLDS.DELTA_ASSIGNMENT_RISK) {
      signal = 'bearish';
      weight = 1.1;
      reasoning = `${(absDelta * 100).toFixed(0)}% delta - high assignment probability`;
    } else if (absDelta <= 0.20) {
      signal = 'bullish';
      reasoning = `${(absDelta * 100).toFixed(0)}% delta - likely expires worthless`;
    } else {
      signal = 'neutral';
      reasoning = `${(absDelta * 100).toFixed(0)}% delta`;
    }
  } else {
    // LONG: Low delta = needs big move (lottery ticket)
    if (absDelta <= THRESHOLDS.DELTA_LOTTERY_TICKET) {
      signal = 'bearish';
      reasoning = `${(absDelta * 100).toFixed(0)}% delta - needs major move`;
    } else if (absDelta >= 0.70) {
      signal = 'bullish';
      reasoning = `${(absDelta * 100).toFixed(0)}% delta - behaves like stock`;
    } else {
      signal = 'neutral';
      reasoning = `${(absDelta * 100).toFixed(0)}% delta`;
    }
  }

  return {
    name: 'Delta',
    value: `${(delta * 100).toFixed(0)}%`,
    signal,
    weight,
    reasoning,
    category: 'greeks',
  };
}

function analyzeGammaRisk(gamma: number, dte: number, isLong: boolean): FactorAnalysis {
  let signal: SignalType = 'neutral';
  let reasoning: string;
  let weight = 0.8;

  const isHighGamma = gamma >= THRESHOLDS.GAMMA_DANGEROUS;
  const nearExpiry = dte <= 14;

  if (isHighGamma && nearExpiry) {
    weight = 1.2;
    if (!isLong) {
      signal = 'bearish';
      reasoning = `Gamma ${gamma.toFixed(3)} + ${dte}d DTE = price swing danger`;
    } else {
      signal = 'neutral';
      reasoning = `High gamma ${gamma.toFixed(3)} - big move could pay off`;
    }
  } else if (isHighGamma) {
    signal = 'neutral';
    reasoning = `Elevated gamma ${gamma.toFixed(3)} - delta may shift quickly`;
  } else {
    signal = 'neutral';
    reasoning = `Gamma ${gamma.toFixed(3)} - stable`;
  }

  return {
    name: 'Gamma',
    value: gamma.toFixed(4),
    signal,
    weight,
    reasoning,
    category: 'greeks',
  };
}

function analyzeThetaEfficiency(theta: number, optionPrice: number, _dte: number): FactorAnalysis {
  // For shorts, theta is income. Calculate daily yield.
  const dailyYield = optionPrice > 0 ? Math.abs(theta) / optionPrice : 0;
  
  let signal: SignalType = 'neutral';
  let reasoning: string;
  const weight = 0.9;

  if (dailyYield >= THRESHOLDS.THETA_EFFICIENT) {
    signal = 'bullish';
    reasoning = `${(dailyYield * 100).toFixed(1)}% daily decay - efficient theta capture`;
  } else if (dailyYield >= 0.015) {
    signal = 'neutral';
    reasoning = `${(dailyYield * 100).toFixed(2)}% daily decay`;
  } else {
    signal = 'bearish';
    reasoning = `${(dailyYield * 100).toFixed(2)}% daily - low theta efficiency`;
  }

  return {
    name: 'Theta Yield',
    value: `$${Math.abs(theta).toFixed(2)}/day`,
    signal,
    weight,
    reasoning,
    category: 'greeks',
  };
}

function analyzeLiquidity(
  bid: number, 
  ask: number, 
  volume: number, 
  openInterest: number
): FactorAnalysis {
  const spread = ask - bid;
  const midPrice = (bid + ask) / 2;
  const spreadPct = midPrice > 0 ? spread / midPrice : 0;

  let signal: SignalType = 'neutral';
  let reasoning: string;
  let weight = 0.8;

  // Priority 1: Check for "Zombie Options" (Ghost Liquidity)
  if (openInterest < THRESHOLDS.OI_ZOMBIE) {
    signal = 'bearish';
    weight = 2.0; // High weight - you might be trapped
    reasoning = `Zombie Option: OI only ${openInterest} - may be trapped`;
  } else if (openInterest < THRESHOLDS.OI_MINIMUM) {
    signal = 'bearish';
    weight = 1.2;
    reasoning = `Low OI (${openInterest}) - exit may be difficult`;
  }
  // Priority 2: Check spread (only if OI is acceptable)
  else if (spreadPct >= THRESHOLDS.SPREAD_ILLIQUID) {
    signal = 'bearish';
    weight = 1.0;
    reasoning = `${(spreadPct * 100).toFixed(0)}% spread - very illiquid`;
  } else if (spreadPct >= THRESHOLDS.SPREAD_WIDE) {
    signal = 'bearish';
    weight = 0.8;
    reasoning = `${(spreadPct * 100).toFixed(0)}% spread - wide, costly exit`;
  }
  // Priority 3: Check for healthy volume
  else if (volume > 0 && volume >= openInterest * THRESHOLDS.VOLUME_HIGH_RATIO) {
    signal = 'bullish';
    reasoning = `High volume (${volume}) vs OI (${openInterest}) - easy execution`;
  } else if (openInterest >= THRESHOLDS.OI_HEALTHY && spreadPct <= 0.03) {
    signal = 'bullish';
    reasoning = `Liquid: OI ${openInterest}, ${(spreadPct * 100).toFixed(1)}% spread`;
  } else if (volume === 0 && openInterest >= THRESHOLDS.OI_HEALTHY) {
    signal = 'neutral';
    reasoning = `No volume today, but OI ${openInterest} suggests liquidity`;
  } else {
    signal = 'neutral';
    reasoning = `OI ${openInterest}, ${(spreadPct * 100).toFixed(1)}% spread - acceptable`;
  }

  return {
    name: 'Liquidity Health',
    value: `OI: ${openInterest} | Vol: ${volume}`,
    signal,
    weight,
    reasoning,
    category: 'structure',
  };
}

function analyzeMoneyness(
  underlyingPrice: number,
  strikePrice: number,
  isCall: boolean,
  isLong: boolean,
  dte: number
): FactorAnalysis {
  // Calculate how far ITM or OTM as a percentage
  const moneyness = isCall
    ? (underlyingPrice - strikePrice) / strikePrice
    : (strikePrice - underlyingPrice) / strikePrice;

  const isItm = moneyness > 0;
  const absMoneyness = Math.abs(moneyness);

  let signal: SignalType = 'neutral';
  let reasoning: string;
  let weight = 0.7;

  const label = isItm
    ? absMoneyness > 0.15 ? 'Deep ITM' : 'ITM'
    : absMoneyness > 0.15 ? 'Deep OTM' : absMoneyness < 0.02 ? 'ATM' : 'OTM';

  if (isLong) {
    if (!isItm && absMoneyness > 0.15 && dte <= 14) {
      signal = 'bearish';
      weight = 1.0;
      reasoning = `${label} (${(absMoneyness * 100).toFixed(0)}%) with ${dte}d - low recovery odds`;
    } else if (isItm && absMoneyness > 0.10) {
      signal = 'bullish';
      reasoning = `${label} (${(absMoneyness * 100).toFixed(0)}%) - strong position`;
    } else {
      signal = 'neutral';
      reasoning = `${label} - ${(moneyness * 100).toFixed(1)}% from strike`;
    }
  } else {
    // Short positions
    if (isItm && absMoneyness > 0.10) {
      signal = 'bearish';
      weight = 1.0;
      reasoning = `${label} (${(absMoneyness * 100).toFixed(0)}%) - assignment risk`;
    } else if (!isItm && absMoneyness > 0.10) {
      signal = 'bullish';
      reasoning = `${label} (${(absMoneyness * 100).toFixed(0)}%) - safe from strike`;
    } else {
      signal = 'neutral';
      reasoning = `${label}`;
    }
  }

  return {
    name: 'Moneyness',
    value: label,
    signal,
    weight,
    reasoning,
    category: 'structure',
  };
}

function analyzeEarningsRisk(
  dte: number, 
  earningsDateInput: string | Date, 
  isLong: boolean
): FactorAnalysis {
  const earningsDate = new Date(earningsDateInput);
  const today = new Date();
  // Reset time portion for accurate day calculation
  today.setHours(0, 0, 0, 0);
  earningsDate.setHours(0, 0, 0, 0);
  
  const daysToEarnings = Math.ceil((earningsDate.getTime() - today.getTime()) / (1000 * 3600 * 24));

  let signal: SignalType = 'neutral';
  let reasoning: string;
  let weight = 1.0;

  // Earnings already passed
  if (daysToEarnings < 0) {
    signal = 'neutral';
    reasoning = 'Earnings passed - IV likely crushed';
    weight = 0.5;
  }
  // Earnings happening BEFORE expiration (within DTE)
  else if (daysToEarnings <= dte) {
    // This is the critical zone - binary event risk
    if (daysToEarnings <= THRESHOLDS.EARNINGS_DANGER_ZONE) {
      signal = 'bearish';
      weight = 3.0; // Massive weight - overrides most technicals
      if (isLong) {
        reasoning = `EARNINGS in ${daysToEarnings}d! IV Crush will destroy time value`;
      } else {
        reasoning = `EARNINGS in ${daysToEarnings}d! Gap risk bypasses stop loss`;
      }
    } else if (daysToEarnings <= THRESHOLDS.EARNINGS_WARNING) {
      signal = 'bearish';
      weight = 2.0;
      reasoning = `Earnings in ${daysToEarnings}d (inside DTE) - Binary event risk`;
    } else {
      signal = 'bearish';
      weight = 1.5;
      reasoning = `Earnings in ${daysToEarnings}d before expiry - elevated risk`;
    }
  }
  // Earnings after expiration - clear runway
  else {
    signal = 'bullish';
    reasoning = `Earnings in ${daysToEarnings}d (after expiry) - clear runway`;
    weight = 0.8;
  }

  return {
    name: 'Earnings Risk',
    value: `${daysToEarnings}d`,
    signal,
    weight,
    reasoning,
    category: 'market',
  };
}

function analyzeRSI(rsi: number, isLong: boolean, isCall: boolean): FactorAnalysis {
  let signal: SignalType = 'neutral';
  let reasoning: string;
  let weight = 1.0;

  // Determine position strategy
  const isBullishPosition = (isCall && isLong) || (!isCall && !isLong); // Long Call or Short Put
  const isBearishPosition = (!isCall && isLong) || (isCall && !isLong); // Long Put or Short Call

  if (isBullishPosition) {
    // Bullish positions benefit from oversold conditions (mean reversion up)
    if (rsi <= THRESHOLDS.RSI_EXTREME_LOW) {
      signal = 'bullish';
      weight = 1.8;
      reasoning = `RSI ${rsi} (Extreme Oversold) - Ideal for bullish position`;
    } else if (rsi <= THRESHOLDS.RSI_OVERSOLD) {
      signal = 'bullish';
      weight = 1.5;
      reasoning = `RSI ${rsi} (Oversold) - Good entry zone for bullish`;
    } else if (rsi >= THRESHOLDS.RSI_EXTREME_HIGH) {
      signal = 'bearish';
      weight = 1.6;
      reasoning = `RSI ${rsi} (Extreme Overbought) - Pullback risk for bullish position`;
    } else if (rsi >= THRESHOLDS.RSI_OVERBOUGHT) {
      signal = 'bearish';
      weight = 1.3;
      reasoning = `RSI ${rsi} (Overbought) - Consider taking profits`;
    } else {
      signal = 'neutral';
      reasoning = `RSI ${rsi} - Neutral zone`;
    }
  } else if (isBearishPosition) {
    // Bearish positions benefit from overbought conditions (mean reversion down)
    if (rsi >= THRESHOLDS.RSI_EXTREME_HIGH) {
      signal = 'bullish';
      weight = 1.8;
      reasoning = `RSI ${rsi} (Extreme Overbought) - Ideal for bearish position`;
    } else if (rsi >= THRESHOLDS.RSI_OVERBOUGHT) {
      signal = 'bullish';
      weight = 1.5;
      reasoning = `RSI ${rsi} (Overbought) - Good entry zone for bearish`;
    } else if (rsi <= THRESHOLDS.RSI_EXTREME_LOW) {
      signal = 'bearish';
      weight = 1.6;
      reasoning = `RSI ${rsi} (Extreme Oversold) - Bounce risk for bearish position`;
    } else if (rsi <= THRESHOLDS.RSI_OVERSOLD) {
      signal = 'bearish';
      weight = 1.3;
      reasoning = `RSI ${rsi} (Oversold) - Consider taking profits`;
    } else {
      signal = 'neutral';
      reasoning = `RSI ${rsi} - Neutral zone`;
    }
  } else {
    // Fallback for edge cases
    signal = 'neutral';
    reasoning = `RSI ${rsi}`;
  }

  return {
    name: 'RSI Timing',
    value: `${rsi}`,
    signal,
    weight,
    reasoning,
    category: 'market',
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

function getDefaultReason(action: OptionAction, option: OptionHoldingWithData): string {
  const posType = option.position_type === 'long' ? 'Long' : 'Short';
  const optType = option.option_type === 'call' ? 'Call' : 'Put';

  switch (action) {
    case 'CLOSE':
      return `Consider closing this ${posType} ${optType}`;
    case 'OPEN_MORE':
      return `Conditions favor adding to this ${posType} ${optType}`;
    case 'HOLD':
    default:
      return `No strong signal for this ${posType} ${optType}`;
  }
}

function generateSummary(
  action: OptionAction,
  option: OptionHoldingWithData,
  topReason: string
): string {
  const ticker = option.underlying_ticker;
  const strike = option.strike_price;
  const type = option.option_type.toUpperCase();

  return `${action} ${ticker} $${strike} ${type} - ${topReason}`;
}

// ============================================================================
// Batch Analysis
// ============================================================================

export function analyzeOptionPositions(
  options: OptionHoldingWithData[],
  stockAnalysisMap: Map<string, StockAnalysisData>
): Map<number, OptionRecommendation> {
  const results = new Map<number, OptionRecommendation>();

  for (const option of options) {
    const stockAnalysis = stockAnalysisMap.get(option.underlying_ticker);
    const recommendation = analyzeOptionPosition(option, stockAnalysis);
    results.set(option.id, recommendation);
  }

  return results;
}
