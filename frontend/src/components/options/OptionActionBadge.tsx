/**
 * Option Action Badge
 * 
 * Displays a small badge with the recommended action for an option position.
 * Color-coded based on recommendation:
 * - CLOSE: Red/orange
 * - HOLD: Neutral gray
 * - OPEN_MORE: Green
 */

import { useMemo } from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import type { OptionHoldingWithData } from '../../types';
import { 
  analyzeOptionPosition, 
  type OptionRecommendation,
  type StockAnalysisData,
  ACTION_COLORS,
  ACTION_LABELS
} from '../../services/optionScoring';

interface OptionActionBadgeProps {
  option: OptionHoldingWithData;
  stockAnalysis?: StockAnalysisData;
  onClick?: () => void;
  showConfidence?: boolean;
}

export function OptionActionBadge({ 
  option, 
  stockAnalysis,
  onClick,
  showConfidence = false
}: OptionActionBadgeProps) {
  // Compute recommendation
  const recommendation = useMemo((): OptionRecommendation => {
    return analyzeOptionPosition(option, stockAnalysis);
  }, [option, stockAnalysis]);
  
  const colors = ACTION_COLORS[recommendation.action];
  const label = ACTION_LABELS[recommendation.action];
  
  // Icon based on action
  const Icon = recommendation.action === 'CLOSE' 
    ? TrendingDown 
    : recommendation.action === 'OPEN_MORE' 
      ? TrendingUp 
      : Minus;
  
  return (
    <button
      onClick={onClick}
      className={`
        inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold
        border transition-all duration-200
        ${colors.bg} ${colors.text} ${colors.border}
        ${onClick ? 'hover:brightness-110 cursor-pointer' : 'cursor-default'}
      `}
      title={`${label}: ${recommendation.summary}`}
    >
      <Icon className="w-3 h-3" />
      <span>{label}</span>
      {showConfidence && (
        <span className="opacity-70">
          {recommendation.confidence}%
        </span>
      )}
    </button>
  );
}

/**
 * Hook to get recommendation for an option
 */
export function useOptionRecommendation(
  option: OptionHoldingWithData,
  stockAnalysis?: StockAnalysisData
): OptionRecommendation {
  return useMemo(() => {
    return analyzeOptionPosition(option, stockAnalysis);
  }, [option, stockAnalysis]);
}

