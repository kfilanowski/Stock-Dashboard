import { useState, useEffect, useMemo, useRef } from 'react';
import { Plus, Briefcase } from 'lucide-react';
import { 
  Header, 
  PortfolioSummary, 
  HoldingCard, 
  AddHoldingModal, 
  PortfolioSettings,
  StockDetailModal,
  ChartPeriodSelector,
  RefreshTimer,
  type ChartPeriod
} from './components';
import { usePortfolio } from './hooks/usePortfolio';
import type { HistoryPoint } from './types';
import * as api from './services/api';

interface HoldingChartData {
  history: HistoryPoint[];
  referenceClose: number | null;
  isComplete: boolean;
  expectedStart: string | null;
  actualStart: string | null;
}

const REFRESH_INTERVAL = 10000; // 10 seconds

function App() {
  const { 
    portfolio, 
    loading, 
    refreshing,
    error, 
    lastFetched,
    refresh, 
    updatePortfolioValue, 
    addHolding, 
    removeHolding,
    updateHolding 
  } = usePortfolio(REFRESH_INTERVAL);

  const [showAddModal, setShowAddModal] = useState(false);
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const [holdingChartData, setHoldingChartData] = useState<Record<string, HoldingChartData>>({});
  const [chartPeriod, setChartPeriod] = useState<ChartPeriod>('1d');
  const [loadingHistories, setLoadingHistories] = useState(false);
  
  // Track tickers to only refetch when tickers actually change
  const tickersKey = useMemo(() => {
    return portfolio?.holdings.map(h => h.ticker).sort().join(',') ?? '';
  }, [portfolio?.holdings]);
  
  const prevTickersRef = useRef(tickersKey);
  const prevPeriodRef = useRef(chartPeriod);

  // Fetch histories only when period or tickers change (not on every portfolio refresh)
  useEffect(() => {
    const tickersChanged = prevTickersRef.current !== tickersKey;
    const periodChanged = prevPeriodRef.current !== chartPeriod;
    
    prevTickersRef.current = tickersKey;
    prevPeriodRef.current = chartPeriod;
    
    // Only fetch if period or tickers changed
    if (!tickersChanged && !periodChanged && Object.keys(holdingChartData).length > 0) {
      return;
    }
    
    if (!portfolio?.holdings.length) {
      setHoldingChartData({});
      return;
    }

    const fetchHistories = async () => {
      setLoadingHistories(true);
      const chartData: Record<string, HoldingChartData> = {};
      
      await Promise.all(
        portfolio.holdings.map(async (holding) => {
          try {
            const data = await api.getStockHistory(holding.ticker, chartPeriod);
            chartData[holding.ticker] = {
              history: data.history,
              referenceClose: data.reference_close,
              isComplete: data.is_complete,
              expectedStart: data.expected_start,
              actualStart: data.actual_start
            };
          } catch {
            chartData[holding.ticker] = { 
              history: [], 
              referenceClose: null, 
              isComplete: false,
              expectedStart: null,
              actualStart: null
            };
          }
        })
      );
      
      setHoldingChartData(chartData);
      setLoadingHistories(false);
    };

    fetchHistories();
  }, [tickersKey, chartPeriod, portfolio?.holdings]);

  const handleAddHolding = async (ticker: string, allocation: number) => {
    await addHolding(ticker, allocation);
  };

  const handleUpdateAllocation = async (holdingId: number, allocation: number) => {
    await updateHolding(holdingId, allocation);
  };

  const currentAllocation = portfolio?.holdings.reduce((sum, h) => sum + h.allocation_pct, 0) ?? 0;

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
          totalValue={portfolio?.total_value ?? 0}
          onRefresh={refresh}
          loading={loading || refreshing}
          lastUpdated={lastFetched ?? undefined}
        />

        {portfolio && (
          <>
            <PortfolioSummary 
              portfolio={portfolio} 
              isRefreshing={refreshing} 
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
                  <span className="text-white/20">•</span>
                  <RefreshTimer 
                    lastFetched={lastFetched} 
                    refreshInterval={REFRESH_INTERVAL}
                    isRefreshing={refreshing}
                  />
                </div>
                
                <div className="flex items-center gap-3">
                  <PortfolioSettings 
                    currentValue={portfolio.total_value}
                    onUpdate={updatePortfolioValue}
                  />
                  <button 
                    onClick={() => setShowAddModal(true)}
                    className="btn-primary flex items-center gap-2"
                  >
                    <Plus className="w-5 h-5" />
                    Add Stock
                  </button>
                </div>
              </div>

              {/* Chart Period Selector */}
              {portfolio.holdings.length > 0 && (
                <div className="mb-4 flex items-center gap-3">
                  <span className="text-white/50 text-sm">Chart period:</span>
                  <ChartPeriodSelector 
                    selected={chartPeriod} 
                    onSelect={setChartPeriod} 
                  />
                  {loadingHistories && (
                    <div className="w-4 h-4 border-2 border-white/20 border-t-accent-cyan rounded-full animate-spin" />
                  )}
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
                  {portfolio.holdings.map((holding, index) => {
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
                          currentTotalAllocation={currentAllocation}
                          isRefreshing={refreshing}
                          isHistoryLoading={loadingHistories}
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
      />

      <StockDetailModal
        ticker={selectedTicker}
        onClose={() => setSelectedTicker(null)}
      />
    </div>
  );
}

export default App;
