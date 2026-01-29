import { useMemo } from 'react';
import type { HistoryPoint } from '../../types';
import type { ChartPeriod } from '../ChartPeriodSelector';

interface PeriodHighLowProps {
  history: HistoryPoint[];
  currentPrice?: number;
  period: ChartPeriod;
}

// Convert ChartPeriod to display label
const PERIOD_LABELS: Record<ChartPeriod, string> = {
  '1d': '1 Day',
  '3d': '3 Day',
  '1w': '1 Week',
  '1mo': '1 Month',
  '3mo': '3 Month',
  '6mo': '6 Month',
  'ytd': 'YTD',
  '1y': '1 Year',
  '2y': '2 Year',
  '5y': '5 Year',
};

export function PeriodHighLow({ history, currentPrice, period }: PeriodHighLowProps) {
  const { periodHigh, periodLow } = useMemo(() => {
    if (!history.length) {
      return { periodHigh: null, periodLow: null };
    }

    // Calculate high/low from all candles in the period
    // For intraday data, each point represents a time interval with its own high/low
    // For daily data, each point is a day with its high/low
    let high = -Infinity;
    let low = Infinity;

    for (const point of history) {
      if (point.high > high) high = point.high;
      if (point.low < low) low = point.low;
    }

    return {
      periodHigh: high === -Infinity ? null : high,
      periodLow: low === Infinity ? null : low
    };
  }, [history]);

  if (periodHigh === null || periodLow === null) {
    return null;
  }

  // Calculate percentage from current price to high/low
  const highPct = currentPrice ? ((periodHigh - currentPrice) / currentPrice) * 100 : null;
  const lowPct = currentPrice ? ((currentPrice - periodLow) / currentPrice) * 100 : null;

  const periodLabel = PERIOD_LABELS[period] || period;

  return (
    <div className="flex items-center gap-3 text-[13px]">
      {/* Period Label */}
      <span className="text-white/40 font-medium">{periodLabel}</span>
      <span className="text-white/20">—</span>

      {/* Period High */}
      <div className="flex items-center gap-1.5" title={`Period High: $${periodHigh.toFixed(2)}${highPct !== null ? ` (${highPct >= 0 ? '+' : ''}${highPct.toFixed(1)}% from current)` : ''}`}>
        <span className="text-white/50">High:</span>
        <span className="text-green-400 font-medium">${periodHigh.toFixed(2)}</span>
        {highPct !== null && highPct > 0 && (
          <span className="text-green-400/70 text-xs">+{highPct.toFixed(1)}%</span>
        )}
      </div>

      {/* Period Low */}
      <div className="flex items-center gap-1.5" title={`Period Low: $${periodLow.toFixed(2)}${lowPct !== null ? ` (${lowPct >= 0 ? '-' : ''}${Math.abs(lowPct).toFixed(1)}% from current)` : ''}`}>
        <span className="text-white/50">Low:</span>
        <span className="text-red-400 font-medium">${periodLow.toFixed(2)}</span>
        {lowPct !== null && lowPct > 0 && (
          <span className="text-red-400/70 text-xs">-{lowPct.toFixed(1)}%</span>
        )}
      </div>
    </div>
  );
}
