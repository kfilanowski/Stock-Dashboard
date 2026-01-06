import { useMemo } from 'react';
import { Trash2, BarChart3, RefreshCw, TrendingUp, TrendingDown } from 'lucide-react';
import type { Holding, HistoryPoint } from '../types';
import { AllocationEditor, InvestmentInfoEditor, MiniStockChart } from './holding';

interface HoldingCardProps {
  holding: Holding;
  history: HistoryPoint[];
  referenceClose: number | null;
  isDataComplete?: boolean;
  expectedStart?: string | null;
  actualStart?: string | null;
  onDelete: (id: number) => void;
  onSelect: (ticker: string) => void;
  onUpdateAllocation: (id: number, allocation: number) => Promise<void>;
  onUpdateInvestment: (id: number, data: { investment_date?: string; investment_price?: number }) => Promise<void>;
  currentTotalAllocation: number;
  portfolioTotalValue: number;
  isRefreshing?: boolean;
  isHistoryLoading?: boolean;
  lastPricesFetched?: Date | null;
}

export function HoldingCard({ 
  holding, 
  history, 
  referenceClose,
  isDataComplete = true,
  expectedStart,
  actualStart,
  onDelete, 
  onSelect, 
  onUpdateAllocation,
  onUpdateInvestment,
  currentTotalAllocation,
  portfolioTotalValue,
  isRefreshing = false,
  isHistoryLoading = false,
  lastPricesFetched
}: HoldingCardProps) {
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
          
          {/* Allocation, Value, and YTD stacked */}
          <div className="text-right">
            <AllocationEditor
              holdingId={holding.id}
              currentAllocation={holding.allocation_pct}
              portfolioTotalValue={portfolioTotalValue}
              currentTotalAllocation={currentTotalAllocation}
              onSave={onUpdateAllocation}
            />
            <p className="text-white font-medium text-sm">
              ${holding.current_value?.toLocaleString('en-US', { minimumFractionDigits: 2 }) ?? '—'}
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

      {/* Investment info */}
      <InvestmentInfoEditor
        holdingId={holding.id}
        investmentDate={holding.investment_date}
        investmentPrice={holding.investment_price}
        currentPrice={holding.current_price}
        gainLossPct={holding.gain_loss_pct}
        onSave={onUpdateInvestment}
      />

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
