import { useMemo } from 'react';
import { TrendingUp, TrendingDown, RefreshCw } from 'lucide-react';
import type { Holding, HistoryPoint } from '../../types';

interface HoldingMetricsProps {
  holding: Holding;
  history: HistoryPoint[];
  referenceClose: number | null;
  isDataComplete?: boolean;
  isRefreshing?: boolean;
}

function UpdateIndicator({ updating }: { updating: boolean }) {
  if (!updating) return null;
  return (
    <RefreshCw className="w-3 h-3 text-accent-cyan/60 animate-spin inline ml-1" />
  );
}

export function HoldingMetrics({
  holding,
  history,
  referenceClose,
  isDataComplete = true,
  isRefreshing = false
}: HoldingMetricsProps) {
  const isPositive = (holding.ytd_return ?? 0) >= 0;

  // Calculate period gain
  const periodGain = useMemo(() => {
    if (!history.length || referenceClose === null || referenceClose === 0) return null;
    const latestClose = history[history.length - 1]?.close ?? 0;
    return ((latestClose - referenceClose) / referenceClose) * 100;
  }, [history, referenceClose]);

  const isAboveReference = periodGain !== null ? periodGain >= 0 : isPositive;

  return (
    <div className="grid grid-cols-2 gap-4 mb-4">
      <div>
        <p className="text-white/50 text-xs mb-1">Current Price</p>
        <p className="text-white font-semibold flex items-center">
          ${holding.current_price?.toFixed(2) ?? '—'}
          <UpdateIndicator updating={isRefreshing} />
        </p>
      </div>
      <div>
        <p className="text-white/50 text-xs mb-1">Value</p>
        <p className="text-white font-semibold">
          ${holding.current_value?.toLocaleString('en-US', { minimumFractionDigits: 2 }) ?? '—'}
          <UpdateIndicator updating={isRefreshing} />
        </p>
      </div>
      <div>
        <p className="text-white/50 text-xs mb-1">YTD Return</p>
        <div className="flex items-center gap-1">
          {isPositive ? (
            <TrendingUp className="w-4 h-4 text-green-400" />
          ) : (
            <TrendingDown className="w-4 h-4 text-red-400" />
          )}
          <p className={`font-semibold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
            {isPositive ? '+' : ''}{holding.ytd_return?.toFixed(2) ?? 0}%
          </p>
          <UpdateIndicator updating={isRefreshing} />
        </div>
      </div>
      <div>
        <p className="text-white/50 text-xs mb-1">Period Gain</p>
        {!isDataComplete ? (
          <p className="text-yellow-500/70 font-semibold text-xs" title="Incomplete historical data">
            Loading...
          </p>
        ) : periodGain !== null ? (
          <div className="flex items-center gap-1">
            {isAboveReference ? (
              <TrendingUp className="w-4 h-4 text-green-400" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-400" />
            )}
            <p className={`font-semibold ${isAboveReference ? 'text-green-400' : 'text-red-400'}`}>
              {periodGain >= 0 ? '+' : ''}{periodGain.toFixed(2)}%
            </p>
          </div>
        ) : (
          <p className="text-white/50 font-semibold">—</p>
        )}
      </div>
    </div>
  );
}

