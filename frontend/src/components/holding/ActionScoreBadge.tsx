/**
 * Action Score Badge
 * 
 * Compact badge component showing the best recommended action
 * for display on holding cards.
 */

import { useMemo } from 'react';
import { Zap } from 'lucide-react';
import type { HistoryPoint, ActionType } from '../../types';
import { getQuickAnalysis } from '../../services/stockScoring';

interface ActionScoreBadgeProps {
  history: HistoryPoint[];
  currentPrice?: number;
  high52w?: number | null;
  low52w?: number | null;
  onClick?: () => void;
}

// Short labels for compact display
const SHORT_LABELS: Record<ActionType, string> = {
  buyShares: 'Buy',
  sellShares: 'Sell',
  buyCSP: 'CSP',
  buyCC: 'CC',
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
  history,
  currentPrice,
  high52w,
  low52w,
  onClick
}: ActionScoreBadgeProps) {
  const analysis = useMemo(() => {
    if (!history?.length || !currentPrice) return null;
    return getQuickAnalysis(history, currentPrice, high52w ?? null, low52w ?? null);
  }, [history, currentPrice, high52w, low52w]);
  
  if (!analysis) {
    return null;
  }
  
  const styles = getScoreStyles(analysis.score);
  const shortLabel = SHORT_LABELS[analysis.action];
  
  return (
    <button
      onClick={onClick}
      className={`
        inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium
        border transition-all hover:scale-105 active:scale-95
        ${styles.bg} ${styles.text} ${styles.border}
      `}
      title={`${analysis.label}: ${analysis.score}/100 (${analysis.confidence} confidence)`}
    >
      <Zap className="w-3 h-3" />
      <span>{shortLabel}</span>
      <span className="opacity-70">{analysis.score}</span>
    </button>
  );
}

