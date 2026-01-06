import { Trash2, BarChart3, RefreshCw } from 'lucide-react';
import type { Holding, HistoryPoint } from '../types';
import { AllocationEditor, InvestmentInfoEditor, MiniStockChart, HoldingMetrics } from './holding';

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
  return (
    <div className="glass-card p-5 hover:border-white/20 transition-all duration-300 group">
      {/* Header with ticker and actions */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-accent-cyan/30 to-accent-purple/30 flex items-center justify-center border border-white/10">
            <span className="font-bold text-white text-sm">{holding.ticker.slice(0, 3)}</span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-bold text-white text-lg">{holding.ticker}</h3>
              {isRefreshing && <RefreshCw className="w-3 h-3 text-accent-cyan/60 animate-spin" />}
            </div>
            <AllocationEditor
              holdingId={holding.id}
              currentAllocation={holding.allocation_pct}
              portfolioTotalValue={portfolioTotalValue}
              currentTotalAllocation={currentTotalAllocation}
              onSave={onUpdateAllocation}
            />
          </div>
        </div>
        
        <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={() => onSelect(holding.ticker)}
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
            title="View details"
          >
            <BarChart3 className="w-4 h-4 text-white/70" />
          </button>
          <button
            onClick={() => onDelete(holding.id)}
            className="p-2 rounded-lg bg-red-500/10 hover:bg-red-500/20 transition-colors"
            title="Remove holding"
          >
            <Trash2 className="w-4 h-4 text-red-400" />
          </button>
        </div>
      </div>

      {/* Metrics grid */}
      <HoldingMetrics
        holding={holding}
        history={history}
        referenceClose={referenceClose}
        isDataComplete={isDataComplete}
        isRefreshing={isRefreshing}
      />

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
