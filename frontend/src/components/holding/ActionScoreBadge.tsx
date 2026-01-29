/**
 * Action Score Badge Components
 *
 * ActionScoreBadge: Single-horizon badge (legacy, used by modal)
 * DualActionScoreBadge: Combined badge showing both swing (3d) and trend (15d) horizons
 *
 * Uses background analysis with caching - shows loading spinner
 * while analysis is being calculated, then displays the score.
 */

import { useState, useEffect } from 'react';
import { Calculator, Zap, AlertCircle, Clock, Shield, AlertTriangle, Minus } from 'lucide-react';
import type { ActionType } from '../../types';
import { useStockAnalysis, useDualHorizonAnalysis, type AnalysisState } from '../../hooks/useStockAnalysis';
import { useViewContext } from '../../context/ViewContext'; // Import view context
import { getRegimeDescription } from '../../services/regimeRules';
import { verifyCalibration } from '../../services/calibrationApi';

interface ActionScoreBadgeProps {
  ticker: string;
  currentPrice?: number;
  high52w?: number | null;
  low52w?: number | null;
  onClick?: () => void;
}

// SQN Quality Color Coding
function getSqnColor(sqn: number | null | undefined): { text: string; label: string } {
  if (sqn === null || sqn === undefined) {
    return { text: 'text-gray-400', label: 'Uncalibrated' };
  }
  if (sqn >= 2.5) return { text: 'text-green-400', label: 'Excellent' };
  if (sqn >= 1.5) return { text: 'text-blue-400', label: 'Good' };
  if (sqn >= 0.5) return { text: 'text-yellow-400', label: 'Fair' };
  return { text: 'text-red-400', label: 'Poor' };
}

// Short labels for compact display
const SHORT_LABELS: Record<ActionType, string> = {
  buyShares: 'Buy',
  sellShares: 'Sell',
  openCSP: 'CSP',
  openCC: 'CC',
  buyCall: 'Call',
  buyPut: 'Put'
};

// Color based on score
function getScoreStyles(score: number): { bg: string; text: string; border: string } {
  if (score >= 70) {
    return { bg: 'bg-green-500/15', text: 'text-green-400', border: 'border-green-500/30' };
  }
  if (score >= 55) {
    return { bg: 'bg-lime-500/15', text: 'text-lime-400', border: 'border-lime-500/30' };
  }
  if (score >= 45) {
    return { bg: 'bg-yellow-500/15', text: 'text-yellow-400', border: 'border-yellow-500/30' };
  }
  if (score >= 30) {
    return { bg: 'bg-orange-500/15', text: 'text-orange-400', border: 'border-orange-500/30' };
  }
  return { bg: 'bg-red-500/15', text: 'text-red-400', border: 'border-red-500/30' };
}

export function ActionScoreBadge({
  ticker,
  currentPrice,
  high52w,
  low52w,
  onClick
}: ActionScoreBadgeProps) {
  const { predictionHorizon } = useViewContext(); // Use the global horizon

  const { analysis, isLoading, isStale, error } = useStockAnalysis(
    ticker,
    currentPrice,
    high52w,
    low52w,
    predictionHorizon // Pass horizon to hook
  );
  
  // Loading state - show calculator icon with spinning animation
  if (isLoading && !analysis) {
    return (
      <div
        className="
          inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium
          border bg-white/5 text-white/40 border-white/10
        "
        title="Calculating analysis..."
      >
        <Calculator className="w-3 h-3 animate-pulse" />
        <span className="opacity-60">...</span>
      </div>
    );
  }

  // Handle LOW CONFIDENCE explicitly
  if (analysis?.status === 'low_confidence') {
       return (
        <button
            onClick={onClick}
            className="
            inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium
            border bg-amber-500/10 text-amber-400/60 border-amber-500/20
            hover:bg-amber-500/15 transition-colors
            "
            title={`Prediction unreliable (SQN < 2.0). The model is not confident in this ${predictionHorizon}d forecast.`}
        >
            <Shield className="w-3 h-3 text-amber-500/50" />
            <span>Low Conf</span>
        </button>
       );
  }

  // Handle REGIME BLOCKED - matches backend WFO hard filter
  if (analysis?.status === 'regime_blocked') {
       return (
        <button
            onClick={onClick}
            className="
            inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium
            border bg-orange-500/10 text-orange-400/60 border-orange-500/20
            hover:bg-orange-500/15 transition-colors
            "
            title={`Regime Risk: ${analysis.marketRegime || 'Volatile market'}. Best action blocked to match backend safety filter.`}
        >
            <AlertTriangle className="w-3 h-3 text-orange-500/60" />
            <span>Regime Risk</span>
        </button>
       );
  }

  // Error state - show error icon
  if (error && !analysis) {
    // Legacy support: Check for low confidence in error string if cache was old
    const isLowConfidence = error.includes("Low calibration confidence") || error.includes("Low confidence");
    
    if (isLowConfidence) {
       return (
        <button
            onClick={onClick}
            className="
            inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium
            border bg-amber-500/10 text-amber-400/60 border-amber-500/20
            hover:bg-amber-500/15 transition-colors
            "
            title={`${error}. Analysis skipped.`}
        >
            <Shield className="w-3 h-3 text-amber-500/50" />
            <span>Low Conf</span>
        </button>
       );
    }

    return (
      <button
        onClick={onClick}
        className="
          inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium
          border bg-red-500/10 text-red-400/60 border-red-500/20
          hover:bg-red-500/15 transition-colors
        "
        title={`Analysis failed: ${error}. Click to retry.`}
      >
        <AlertCircle className="w-3 h-3" />
        <span>—</span>
      </button>
    );
  }
  
  // No analysis available
  if (!analysis) {
    return null;
  }
  
  const bestAction = analysis.bestAction;
  const styles = getScoreStyles(bestAction.totalScore);
  const shortLabel = SHORT_LABELS[bestAction.action];
  const isCalibrated = !!analysis.calibration;
  
  // Build title with enhanced information
  const sqnInfo = getSqnColor(analysis.calibration?.sqn);
  let title = `${bestAction.label} (${predictionHorizon}d): ${bestAction.totalScore}/100 (${bestAction.confidence} confidence)`;

  // Add calibration info with quality indicator
  if (isCalibrated && analysis.calibration?.sqn !== null) {
    title += `\nCalibration: ${sqnInfo.label} (SQN ${analysis.calibration.sqn?.toFixed(2)})`;
  } else if (!isCalibrated) {
    title += '\nUsing default weights (uncalibrated)';
  }

  // Add market regime if available
  if (analysis.marketRegime) {
    title += `\nMarket Regime: ${getRegimeDescription(analysis.marketRegime as any)}`;
  }

  if (isStale) {
    title += '\n[Refreshing...]';
  }
  if (analysis.dataQuality.missingMetrics.length > 0) {
    title += `\nMissing: ${analysis.dataQuality.missingMetrics.slice(0, 3).join(', ')}`;
    if (analysis.dataQuality.missingMetrics.length > 3) {
      title += ` +${analysis.dataQuality.missingMetrics.length - 3} more`;
    }
  }
  
  return (
    <button
      onClick={onClick}
      className={`
        inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium
        border transition-all hover:scale-105 active:scale-95
        ${styles.bg} ${styles.text} ${styles.border}
        ${isStale ? 'opacity-70' : ''}
        ${isCalibrated ? 'ring-1 ring-purple-500/30' : ''}
      `}
      title={title}
    >
      {/* Show clock if stale & refreshing, otherwise Shield (if calibrated) or Zap */}
      {isLoading && isStale ? (
        <Clock className="w-3 h-3 animate-pulse" />
      ) : isCalibrated ? (
        <Shield className="w-3 h-3 text-purple-400" />
      ) : (
        <Zap className="w-3 h-3" />
      )}
      <span>{shortLabel}</span>
      <span className={isStale ? 'opacity-50' : 'opacity-70'}>
        {bestAction.totalScore}
      </span>
    </button>
  );
}

// ============================================================================
// Dual Action Score Badge (Combined Swing + Trend)
// ============================================================================

interface DualActionScoreBadgeProps {
  ticker: string;
  currentPrice?: number;
  high52w?: number | null;
  low52w?: number | null;
  onClick?: () => void;
}

// Low SQN threshold (lowered from 2.8 to 2.0 to avoid flagging decent strategies)
const LOW_SQN_THRESHOLD = 2.0;

// Helper to check if a horizon has low confidence
// Returns true if: status is explicitly 'low_confidence' OR SQN exists and is below threshold
function isLowConfidence(state: AnalysisState): boolean {
  if (state.analysis?.status === 'low_confidence') return true;
  const sqn = state.analysis?.calibration?.sqn;
  // Only flag as low confidence if we HAVE a calibration with a low SQN
  if (sqn !== null && sqn !== undefined && sqn < LOW_SQN_THRESHOLD) {
    return true;
  }
  return false;
}

// Helper to check if the best action is blocked by market regime
// This matches the backend's hard filter behavior during WFO simulation
function isRegimeBlocked(state: AnalysisState): boolean {
  return state.analysis?.status === 'regime_blocked';
}

// Helper to check if a horizon is calibrated (has WFO weights)
function isCalibrated(state: AnalysisState): boolean {
  if (!state.analysis?.calibration) return false;
  // Consider calibrated if sqn exists OR if period is set (weights were applied)
  return state.analysis.calibration.sqn !== null ||
         state.analysis.calibration.period !== undefined;
}

// Helper to get score color class
function getScoreColorClass(score: number): string {
  if (score >= 70) return 'text-green-400';
  if (score >= 55) return 'text-lime-400';
  if (score >= 45) return 'text-yellow-400';
  if (score >= 30) return 'text-orange-400';
  return 'text-red-400';
}

// Render a single horizon row for the vertical badge
function HorizonRow({
  label,
  state
}: {
  label: string;
  state: AnalysisState;
}) {
  const { analysis, isLoading, error } = state;
  const lowConf = isLowConfidence(state);
  const regimeBlocked = isRegimeBlocked(state);
  const calibrated = isCalibrated(state);

  // Loading state
  if (isLoading && !analysis) {
    return (
      <div className="flex items-center justify-between gap-2 text-white/40">
        <span className="text-white/50 font-medium">{label}</span>
        <Calculator className="w-3 h-3 animate-pulse" />
      </div>
    );
  }

  // Error state
  if (error && !analysis) {
    return (
      <div className="flex items-center justify-between gap-2">
        <span className="text-white/50 font-medium">{label}</span>
        <span className="text-red-400/60">—</span>
      </div>
    );
  }

  // No analysis
  if (!analysis) {
    return (
      <div className="flex items-center justify-between gap-2">
        <span className="text-white/50 font-medium">{label}</span>
        <span className="text-white/30">—</span>
      </div>
    );
  }

  const bestAction = analysis.bestAction;
  const shortLabel = SHORT_LABELS[bestAction.action];
  const scoreColor = getScoreColorClass(bestAction.totalScore);

  // Determine warning state (regime blocked takes precedence)
  const hasWarning = regimeBlocked || lowConf;

  return (
    <div className="flex items-center justify-between gap-2">
      <div className="flex items-center gap-1">
        {/* Status indicator for this horizon */}
        {regimeBlocked ? (
          <span title={`Regime Risk: ${analysis.marketRegime || 'Volatile market'}`}>
            <AlertTriangle className="w-3 h-3 text-orange-400" />
          </span>
        ) : calibrated && !lowConf ? (
          <Shield className="w-3 h-3 text-purple-400" />
        ) : calibrated && lowConf ? (
          <span title="Low Confidence (SQN < 2.0)">
            <AlertTriangle className="w-3 h-3 text-amber-400" />
          </span>
        ) : (
          <Minus className="w-3 h-3 text-white/20" />
        )}
        <span className="text-white/50 font-medium">{label}</span>
      </div>
      <div className={`flex items-center gap-1 ${hasWarning ? 'opacity-100' : ''}`}>
        {regimeBlocked ? (
           <span className="text-orange-400 text-[10px] font-medium tracking-wide">Risk</span>
        ) : lowConf ? (
           <span className="text-amber-400 text-[10px] font-medium tracking-wide">Low Conf</span>
        ) : (
           <>
             <span className={scoreColor}>{shortLabel}</span>
             <span className={`${scoreColor} opacity-70`}>{bestAction.totalScore}</span>
           </>
        )}
      </div>
    </div>
  );
}

// Cache for calibrated horizons to avoid repeated API calls
const calibratedHorizonsCache = new Map<string, { horizons: Array<{horizon: number; sqn: number | null}>; fetchedAt: number }>();
const HORIZON_CACHE_TTL = 5 * 60 * 1000; // 5 minutes

export function DualActionScoreBadge({
  ticker,
  currentPrice,
  high52w,
  low52w,
  onClick
}: DualActionScoreBadgeProps) {
  // State for calibrated horizons
  const [calibratedHorizons, setCalibratedHorizons] = useState<Array<{horizon: number; sqn: number | null}>>([]);
  const [horizonsLoaded, setHorizonsLoaded] = useState(false);

  // Determine which horizons to display (top 2 by SQN, or defaults)
  const horizonA = calibratedHorizons.length > 0 ? calibratedHorizons[0].horizon : 3;
  const horizonB = calibratedHorizons.length > 1 ? calibratedHorizons[1].horizon : 15;

  // Fetch calibrated horizons on mount
  useEffect(() => {
    if (!ticker) return;

    // Check cache first
    const cached = calibratedHorizonsCache.get(ticker);
    if (cached && Date.now() - cached.fetchedAt < HORIZON_CACHE_TTL) {
      setCalibratedHorizons(cached.horizons);
      setHorizonsLoaded(true);
      return;
    }

    verifyCalibration(ticker)
      .then(res => {
        if (res.verified && res.weights.length > 0) {
          // Group by horizon and get max SQN for each
          const horizonMap = new Map<number, number | null>();
          for (const w of res.weights) {
            const existing = horizonMap.get(w.horizon);
            if (existing === undefined || (w.sqn_score !== null && (existing === null || w.sqn_score > existing))) {
              horizonMap.set(w.horizon, w.sqn_score);
            }
          }
          // Sort by SQN (highest first), then by horizon
          const sorted = Array.from(horizonMap.entries())
            .map(([horizon, sqn]) => ({ horizon, sqn }))
            .sort((a, b) => {
              // If both have SQN, sort by SQN descending
              if (a.sqn !== null && b.sqn !== null) return b.sqn - a.sqn;
              // Put calibrated (non-null SQN) first
              if (a.sqn !== null) return -1;
              if (b.sqn !== null) return 1;
              // Otherwise sort by horizon
              return a.horizon - b.horizon;
            });

          // Cache the result
          calibratedHorizonsCache.set(ticker, { horizons: sorted, fetchedAt: Date.now() });
          setCalibratedHorizons(sorted);
        }
        setHorizonsLoaded(true);
      })
      .catch(() => {
        setHorizonsLoaded(true);
      });
  }, [ticker]);

  const { swing, trend } = useDualHorizonAnalysis(ticker, currentPrice, high52w, low52w, horizonA, horizonB);

  const swingAnalysis = swing.analysis;
  const trendAnalysis = trend.analysis;

  // Check calibration status
  const swingCalibrated = isCalibrated(swing);
  const trendCalibrated = isCalibrated(trend);

  // Check low confidence per horizon
  const swingLowConf = isLowConfidence(swing);
  const trendLowConf = isLowConfidence(trend);
  const bothLowConf = swingLowConf && trendLowConf;

  // Check regime blocked per horizon
  const swingRegimeBlocked = isRegimeBlocked(swing);
  const trendRegimeBlocked = isRegimeBlocked(trend);
  const bothRegimeBlocked = swingRegimeBlocked && trendRegimeBlocked;

  // Both loading
  if ((swing.isLoading && !swingAnalysis && trend.isLoading && !trendAnalysis) || !horizonsLoaded) {
    return (
      <div
        className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium border bg-white/5 text-white/40 border-white/10"
        title="Calculating analysis..."
      >
        <Calculator className="w-3 h-3 animate-pulse" />
        <span className="opacity-60">...</span>
      </div>
    );
  }

  // Both horizons regime blocked - show simplified Regime Risk badge
  if (bothRegimeBlocked && swingAnalysis && trendAnalysis) {
    const regime = swingAnalysis.marketRegime || trendAnalysis.marketRegime || 'Volatile';
    return (
      <button
        onClick={onClick}
        className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium border bg-orange-500/10 text-orange-400/60 border-orange-500/20 hover:bg-orange-500/15 transition-colors"
        title={`Regime Risk: ${regime}. Best actions blocked by backend safety filter.`}
      >
        <AlertTriangle className="w-3 h-3 text-orange-400" />
        <span>Regime Risk</span>
      </button>
    );
  }

  // Both low confidence AND both calibrated - show simplified badge
  if (bothLowConf && swingAnalysis && trendAnalysis && swingCalibrated && trendCalibrated) {
    return (
      <button
        onClick={onClick}
        className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium border bg-amber-500/10 text-amber-400/60 border-amber-500/20 hover:bg-amber-500/15 transition-colors"
        title="Both horizons have low confidence (SQN < 2.0). Predictions may be unreliable."
      >
        <AlertTriangle className="w-3 h-3 text-amber-400" />
        <span>Low Conf</span>
      </button>
    );
  }

  // Build title with SQN info and quality indicators
  const swingSqnInfo = getSqnColor(swingAnalysis?.calibration?.sqn);
  const trendSqnInfo = getSqnColor(trendAnalysis?.calibration?.sqn);

  let title = '';
  if (swingAnalysis) {
    title += `${horizonA}d: ${swingAnalysis.bestAction.label} (${swingAnalysis.bestAction.totalScore}/100)`;
    if (swingCalibrated && swingAnalysis.calibration?.sqn) {
      title += ` - ${swingSqnInfo.label} (SQN ${swingAnalysis.calibration.sqn.toFixed(2)})`;
    } else {
      title += ' - Default weights';
    }
    if (swingRegimeBlocked) title += ' [REGIME RISK]';
    else if (swingLowConf) title += ' [Low Conf]';
  }
  title += '\n';
  if (trendAnalysis) {
    title += `${horizonB}d: ${trendAnalysis.bestAction.label} (${trendAnalysis.bestAction.totalScore}/100)`;
    if (trendCalibrated && trendAnalysis.calibration?.sqn) {
      title += ` - ${trendSqnInfo.label} (SQN ${trendAnalysis.calibration.sqn.toFixed(2)})`;
    } else {
      title += ' - Default weights';
    }
    if (trendRegimeBlocked) title += ' [REGIME RISK]';
    else if (trendLowConf) title += ' [Low Conf]';
  }

  // Add market regime if available from either horizon
  const regime = swingAnalysis?.marketRegime || trendAnalysis?.marketRegime;
  if (regime) {
    title += `\nMarket: ${getRegimeDescription(regime as any)}`;
  }

  return (
    <button
      onClick={onClick}
      className="flex flex-col gap-0.5 px-2 py-1 rounded-md text-xs font-medium border transition-all hover:scale-105 active:scale-95 bg-white/5 border-white/10 hover:border-white/20 min-w-[90px]"
      title={title}
    >
      {/* First horizon row */}
      <HorizonRow label={`${horizonA}d`} state={swing} />

      {/* Divider */}
      <div className="border-t border-white/10 my-0.5" />

      {/* Second horizon row */}
      <HorizonRow label={`${horizonB}d`} state={trend} />
    </button>
  );
}