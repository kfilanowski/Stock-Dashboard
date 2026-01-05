import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { Plus, Briefcase } from 'lucide-react';
import { 
  Header, 
  PortfolioSummary, 
  HoldingCard, 
  AddHoldingModal, 
  StockDetailModal,
  ChartPeriodSelector,
  SortSelector,
  type ChartPeriod,
  type SortOption
} from './components';
import { usePortfolio, type HoldingChartData } from './hooks/usePortfolio';

function App() {
  const [showAddModal, setShowAddModal] = useState(false);
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const [holdingChartData, setHoldingChartData] = useState<Record<string, HoldingChartData>>({});
  const [chartPeriod, setChartPeriod] = useState<ChartPeriod>('1d');
  const [sortOption, setSortOption] = useState<SortOption>({ field: 'allocation', direction: 'desc' });

  // Callback for when the portfolio hook fetches new history data
  const handleHistoryUpdate = useCallback((ticker: string, data: HoldingChartData) => {
    setHoldingChartData(prev => ({
      ...prev,
      [ticker]: data
    }));
  }, []);

  const { 
    portfolio, 
    loading, 
    error,
    lastFetched,
    refresh, 
    updatePortfolioValue, 
    addHolding, 
    removeHolding,
    updateHolding 
  } = usePortfolio({
    chartPeriod,
    onHistoryUpdate: handleHistoryUpdate
  });
  
  // Track tickers to detect changes
  const tickersKey = useMemo(() => {
    return portfolio?.holdings.map(h => h.ticker).sort().join(',') ?? '';
  }, [portfolio?.holdings]);
  
  const prevPeriodRef = useRef(chartPeriod);

  // Reset chart data when period changes
  useEffect(() => {
    if (prevPeriodRef.current !== chartPeriod) {
      setHoldingChartData({});
      prevPeriodRef.current = chartPeriod;
    }
  }, [chartPeriod]);

  // Clean up chart data when tickers change (remove data for deleted stocks)
  useEffect(() => {
    const currentTickers = new Set(portfolio?.holdings.map(h => h.ticker) ?? []);
    setHoldingChartData(prev => {
      const newData: Record<string, HoldingChartData> = {};
      for (const ticker of Object.keys(prev)) {
        if (currentTickers.has(ticker)) {
          newData[ticker] = prev[ticker];
        }
      }
      return newData;
    });
  }, [tickersKey]);

  // Calculate period gain for a holding (% change from reference close)
  const getPeriodGain = useCallback((ticker: string): number | null => {
    const chartData = holdingChartData[ticker];
    if (!chartData?.history?.length || chartData.referenceClose === null || chartData.referenceClose === 0) {
      return null;
    }
    const latestClose = chartData.history[chartData.history.length - 1]?.close ?? 0;
    return ((latestClose - chartData.referenceClose) / chartData.referenceClose) * 100;
  }, [holdingChartData]);

  // Sort holdings based on current sort option
  const sortedHoldings = useMemo(() => {
    if (!portfolio?.holdings) return [];
    
    const holdings = [...portfolio.holdings];
    const { field, direction } = sortOption;
    const multiplier = direction === 'asc' ? 1 : -1;

    holdings.sort((a, b) => {
      let comparison = 0;

      switch (field) {
        case 'ticker':
          comparison = a.ticker.localeCompare(b.ticker);
          break;
        case 'daily_change': {
          // Use absolute value for daily change sorting
          const gainA = getPeriodGain(a.ticker);
          const gainB = getPeriodGain(b.ticker);
          // Put null values at the end
          if (gainA === null && gainB === null) comparison = 0;
          else if (gainA === null) comparison = 1;
          else if (gainB === null) comparison = -1;
          else comparison = Math.abs(gainA) - Math.abs(gainB);
          break;
        }
        case 'allocation':
          comparison = a.allocation_pct - b.allocation_pct;
          break;
        case 'equity':
          comparison = (a.current_value ?? 0) - (b.current_value ?? 0);
          break;
        case 'ytd':
          comparison = (a.ytd_return ?? 0) - (b.ytd_return ?? 0);
          break;
        default:
          comparison = 0;
      }

      return comparison * multiplier;
    });

    return holdings;
  }, [portfolio?.holdings, sortOption, getPeriodGain]);

  const handleAddHolding = async (ticker: string, allocation: number) => {
    await addHolding(ticker, allocation);
  };

  const handleUpdateAllocation = async (holdingId: number, allocation: number) => {
    await updateHolding(holdingId, { allocation_pct: allocation });
  };

  const handleUpdateInvestment = async (holdingId: number, data: { investment_date?: string; investment_price?: number }) => {
    await updateHolding(holdingId, data);
  };

  const currentAllocation = portfolio?.holdings.reduce((sum, h) => sum + h.allocation_pct, 0) ?? 0;

  // Check if all holdings have loaded their stock data (have current_price)
  const allDataLoaded = !portfolio?.holdings.length || 
    portfolio.holdings.every(h => h.current_price !== undefined && h.current_price !== null);

  if (loading && !portfolio) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animated-bg" />
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-white/20 border-t-accent-cyan rounded-full animate-spin mx-auto mb-4" />
          <p className="text-white/50">Loading portfolio...</p>
        </div>
      </div>
    );
  }

  if (error && !portfolio) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animated-bg" />
        <div className="text-center glass-card p-8 max-w-md">
          <p className="text-red-400 mb-4">{error}</p>
          <button onClick={refresh} className="btn-primary">
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6 md:p-8">
      <div className="animated-bg" />
      
      <div className="max-w-7xl mx-auto">
        <Header 
          totalValue={(portfolio?.total_value ?? 0) + (portfolio?.total_gain_loss ?? 0)}
          lastUpdated={lastFetched ?? undefined}
          isDataReady={allDataLoaded}
        />

        {portfolio && (
          <>
            <PortfolioSummary 
              portfolio={portfolio} 
              onUpdateValue={updatePortfolioValue}
            />

            {/* Holdings Section */}
            <div className="fade-in" style={{ animationDelay: '0.3s' }}>
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
                <div className="flex items-center gap-3">
                  <Briefcase className="w-5 h-5 text-accent-cyan" />
                  <h2 className="text-xl font-semibold text-white">Holdings</h2>
                  <span className="text-white/40 text-sm">
                    ({portfolio.holdings.length} stocks)
                  </span>
                </div>
                
                <button 
                    onClick={() => setShowAddModal(true)}
                    className="btn-primary flex items-center gap-2"
                  >
                    <Plus className="w-5 h-5" />
                    Add Stock
                  </button>
              </div>

              {/* Chart Period & Sort Selectors */}
              {portfolio.holdings.length > 0 && (
                <div className="mb-4 flex flex-wrap items-center gap-x-6 gap-y-3">
                  <div className="flex items-center gap-3">
                    <span className="text-white/50 text-sm">Chart period:</span>
                    <ChartPeriodSelector 
                      selected={chartPeriod} 
                      onSelect={setChartPeriod} 
                    />
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-white/50 text-sm">Sort by:</span>
                    <SortSelector
                      value={sortOption}
                      onChange={setSortOption}
                    />
                  </div>
                </div>
              )}

              {portfolio.holdings.length === 0 ? (
                <div className="glass-card p-12 text-center">
                  <Briefcase className="w-16 h-16 text-white/20 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-white mb-2">No holdings yet</h3>
                  <p className="text-white/50 mb-6">
                    Add your first stock to start tracking your portfolio
                  </p>
                  <button 
                    onClick={() => setShowAddModal(true)}
                    className="btn-primary inline-flex items-center gap-2"
                  >
                    <Plus className="w-5 h-5" />
                    Add Your First Stock
                  </button>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {sortedHoldings.map((holding, index) => {
                    const chartData = holdingChartData[holding.ticker];
                    return (
                      <div 
                        key={holding.id} 
                        className="fade-in"
                        style={{ animationDelay: `${0.1 * (index + 1)}s` }}
                      >
                        <HoldingCard
                          holding={holding}
                          history={chartData?.history ?? []}
                          referenceClose={chartData?.referenceClose ?? null}
                          isDataComplete={chartData?.isComplete ?? false}
                          expectedStart={chartData?.expectedStart ?? null}
                          actualStart={chartData?.actualStart ?? null}
                          onDelete={removeHolding}
                          onSelect={setSelectedTicker}
                          onUpdateAllocation={handleUpdateAllocation}
                          onUpdateInvestment={handleUpdateInvestment}
                          currentTotalAllocation={currentAllocation}
                          portfolioTotalValue={portfolio?.total_value ?? 0}
                          isRefreshing={false}
                          isHistoryLoading={!chartData}
                        />
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {/* Modals */}
      <AddHoldingModal
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
        onAdd={handleAddHolding}
        currentAllocation={currentAllocation}
        portfolioTotalValue={portfolio?.total_value ?? 0}
      />

      <StockDetailModal
        ticker={selectedTicker}
        onClose={() => setSelectedTicker(null)}
      />
    </div>
  );
}

export default App;
