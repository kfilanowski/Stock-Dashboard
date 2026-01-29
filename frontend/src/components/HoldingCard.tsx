import { useMemo, useState } from 'react';
import { Trash2, BarChart3, RefreshCw, TrendingUp, TrendingDown, Activity, RotateCcw, Pin } from 'lucide-react';
import type { Holding, HistoryPoint } from '../types';
import { PositionEditor, MiniStockChart, DualActionScoreBadge, PeriodHighLow } from './holding';
import type { ChartPeriod } from './ChartPeriodSelector';

interface HoldingCardProps {
  holding: Holding;
  history: HistoryPoint[];
  referenceClose: number | null;
  chartPeriod: ChartPeriod;
  isDataComplete?: boolean;
  expectedStart?: string | null;
  actualStart?: string | null;
  onDelete: (id: number) => void;
  onSelect: (ticker: string) => void;
  onAnalyze: (ticker: string) => void;
  onUpdatePosition: (id: number, data: { shares?: number; avg_cost?: number }) => Promise<void>;
  onRefreshHistory: (ticker: string) => Promise<void>;
  onTogglePin?: (id: number, isPinned: boolean) => void;
  isRefreshing?: boolean;
  isHistoryLoading?: boolean;
  lastPricesFetched?: Date | null;
  high52w?: number | null;
  low52w?: number | null;
}

export function HoldingCard({
  holding,
  history,
  referenceClose,
  chartPeriod,
  isDataComplete = true,
  expectedStart,
  actualStart,
  onDelete,
  onSelect,
  onAnalyze,
  onUpdatePosition,
  onRefreshHistory,
  onTogglePin,
  isRefreshing = false,
  isHistoryLoading = false,
  lastPricesFetched,
  high52w,
  low52w
}: HoldingCardProps) {
  const [isRefreshingHistory, setIsRefreshingHistory] = useState(false);

  const handleRefreshHistory = async () => {
    setIsRefreshingHistory(true);
    try {
      await onRefreshHistory(holding.ticker);
    } finally {
      setIsRefreshingHistory(false);
    }
  };
  // Calculate period gain for display under price
  const periodGain = useMemo(() => {
    if (!history.length || referenceClose === null || referenceClose === 0) return null;
    const latestClose = history[history.length - 1]?.close ?? 0;
    return ((latestClose - referenceClose) / referenceClose) * 100;
  }, [history, referenceClose]);

  const isPeriodPositive = periodGain !== null ? periodGain >= 0 : true;
  const isYtdPositive = (holding.ytd_return ?? 0) >= 0;

  return (
    <div className="glass-card p-5 hover:border-white/20 transition-all duration-300 group">
      {/* Header with ticker, price, YTD, and actions */}
      <div className="flex items-start justify-between mb-4">
        {/* Left: Name/Price/Period column */}
        <div className="flex flex-col">
          <div className="flex items-center gap-2">
            <h3 className="font-bold text-white text-xl">{holding.ticker}</h3>
            {isRefreshing && <RefreshCw className="w-3 h-3 text-accent-cyan/60 animate-spin" />}
            {/* Dual Action Score Badge - shows both swing (3d) and trend (15d) */}
            {holding.current_price && (
              <DualActionScoreBadge
                ticker={holding.ticker}
                currentPrice={holding.current_price}
                high52w={high52w}
                low52w={low52w}
                onClick={() => onAnalyze(holding.ticker)}
              />
            )}
          </div>
          <p className="text-white font-semibold text-lg">
            ${holding.current_price?.toFixed(2) ?? '—'}
          </p>
          {!isDataComplete ? (
            <span className="text-yellow-500/70 text-xs">Loading...</span>
          ) : periodGain !== null ? (
            <div className="flex items-center gap-1">
              {isPeriodPositive ? (
                <TrendingUp className="w-3 h-3 text-green-400" />
              ) : (
                <TrendingDown className="w-3 h-3 text-red-400" />
              )}
              <span className={`text-sm ${isPeriodPositive ? 'text-green-400' : 'text-red-400'}`}>
                {periodGain >= 0 ? '+' : ''}{periodGain.toFixed(2)}%
              </span>
            </div>
          ) : null}
        </div>

        {/* Right: Actions + Allocation/Value/YTD stacked */}
        <div className="flex flex-col items-end gap-1">
          {/* Action buttons */}
          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            {onTogglePin && (
              <button
                onClick={() => onTogglePin(holding.id, holding.is_pinned ?? false)}
                className={`p-1.5 rounded-lg transition-colors ${
                  holding.is_pinned 
                    ? 'bg-accent-cyan/20 hover:bg-accent-cyan/30' 
                    : 'bg-white/5 hover:bg-white/10'
                }`}
                title={holding.is_pinned ? 'Unpin from top' : 'Pin to top'}
              >
                <Pin className={`w-4 h-4 ${holding.is_pinned ? 'text-accent-cyan' : 'text-white/70'}`} />
              </button>
            )}
            <button
              onClick={handleRefreshHistory}
              disabled={isRefreshingHistory}
              className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 transition-colors disabled:opacity-50"
              title="Refresh chart history"
            >
              <RotateCcw className={`w-4 h-4 text-white/70 ${isRefreshingHistory ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={() => onAnalyze(holding.ticker)}
              className="p-1.5 rounded-lg bg-accent-cyan/10 hover:bg-accent-cyan/20 transition-colors"
              title="Analyze actions"
            >
              <Activity className="w-4 h-4 text-accent-cyan" />
            </button>
            <button
              onClick={() => onSelect(holding.ticker)}
              className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
              title="View details"
            >
              <BarChart3 className="w-4 h-4 text-white/70" />
            </button>
            <button
              onClick={() => onDelete(holding.id)}
              className="p-1.5 rounded-lg bg-red-500/10 hover:bg-red-500/20 transition-colors"
              title="Remove holding"
            >
              <Trash2 className="w-4 h-4 text-red-400" />
            </button>
          </div>
          
          {/* Market Value, Allocation, and YTD stacked */}
          <div className="text-right">
            {/* Allocation percentage */}
            {holding.allocation_pct !== null && holding.allocation_pct !== undefined && (
              <p className="text-white/50 text-sm">{holding.allocation_pct.toFixed(1)}% allocation</p>
            )}
            {/* Market value */}
            <p className="text-white font-medium text-sm">
              ${holding.market_value?.toLocaleString('en-US', { minimumFractionDigits: 2 }) ?? '—'}
            </p>
            <div className="flex items-center gap-1 justify-end mt-0.5">
              <span className="text-white/40 text-xs">YTD</span>
              {isYtdPositive ? (
                <TrendingUp className="w-3 h-3 text-green-400" />
              ) : (
                <TrendingDown className="w-3 h-3 text-red-400" />
              )}
              <span className={`text-xs font-medium ${isYtdPositive ? 'text-green-400' : 'text-red-400'}`}>
                {isYtdPositive ? '+' : ''}{holding.ytd_return?.toFixed(2) ?? 0}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Position info (shares & avg cost) */}
      <PositionEditor
        holdingId={holding.id}
        shares={holding.shares}
        avgCost={holding.avg_cost}
        currentPrice={holding.current_price}
        gainLoss={holding.gain_loss}
        gainLossPct={holding.gain_loss_pct}
        onSave={onUpdatePosition}
      />

      {/* Period High/Low */}
      {history.length > 0 && (
        <div className="flex justify-center mb-2">
          <PeriodHighLow history={history} currentPrice={holding.current_price} period={chartPeriod} />
        </div>
      )}

      {/* Mini chart */}
      <MiniStockChart
        holdingId={holding.id}
        history={history}
        referenceClose={referenceClose}
        isDataComplete={isDataComplete}
        expectedStart={expectedStart}
        actualStart={actualStart}
        isLoading={isHistoryLoading}
        lastPricesFetched={lastPricesFetched}
      />
    </div>
  );
}
