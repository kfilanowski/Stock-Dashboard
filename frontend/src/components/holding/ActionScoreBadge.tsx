/**
 * Action Score Badge
 * 
 * Compact badge component showing the best recommended action.
 * Uses background analysis with caching - shows loading spinner
 * while analysis is being calculated, then displays the score.
 */

import { Calculator, Zap, AlertCircle, Clock } from 'lucide-react';
import type { ActionType } from '../../types';
import { useStockAnalysis } from '../../hooks/useStockAnalysis';

interface ActionScoreBadgeProps {
  ticker: string;
  currentPrice?: number;
  high52w?: number | null;
  low52w?: number | null;
  onClick?: () => void;
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
  const { analysis, isLoading, isStale, error } = useStockAnalysis(
    ticker,
    currentPrice,
    high52w,
    low52w
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
  
  // Error state - show error icon
  if (error && !analysis) {
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
  
  // Build title with stale indicator
  let title = `${bestAction.label}: ${bestAction.totalScore}/100 (${bestAction.confidence} confidence)`;
  if (isStale) {
    title += ' • Refreshing...';
  }
  if (analysis.dataQuality.missingMetrics.length > 0) {
    title += ` • Missing: ${analysis.dataQuality.missingMetrics.slice(0, 3).join(', ')}`;
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
      `}
      title={title}
    >
      {/* Show clock if stale & refreshing, otherwise zap */}
      {isLoading && isStale ? (
        <Clock className="w-3 h-3 animate-pulse" />
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
