import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { Plus, Briefcase, Layers, TrendingUp, RefreshCw } from 'lucide-react';
import { 
  Header, 
  PortfolioSummary, 
  HoldingCard, 
  AddHoldingModal, 
  StockDetailModal,
  StockAnalysisModal,
  StockCompareModal,
  ChartPeriodSelector,
  SortSelector,
  ErrorBoundary,
  SectionErrorBoundary,
  OptionHoldingCard,
  AddOptionModal,
  type ChartPeriod,
  type SortOption
} from './components';
import { usePortfolio, type HoldingChartData } from './hooks/usePortfolio';
import { getCachedAnalysisScore } from './hooks/useStockAnalysis';
import * as api from './services/api';
import type { OptionHoldingWithData, OptionHoldingCreate } from './types';

function App() {
  const [showAddModal, setShowAddModal] = useState(false);
  const [showAddOptionModal, setShowAddOptionModal] = useState(false);
  const [showCompareModal, setShowCompareModal] = useState(false);
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const [analyzeTicker, setAnalyzeTicker] = useState<string | null>(null);
  const [holdingChartData, setHoldingChartData] = useState<Record<string, HoldingChartData>>({});
  const [chartPeriod, setChartPeriod] = useState<ChartPeriod>('1d');
  const [sortOption, setSortOption] = useState<SortOption>({ field: 'allocation', direction: 'desc' });
  const [settingsInitialized, setSettingsInitialized] = useState(false);
  
  // Options state
  const [optionHoldings, setOptionHoldings] = useState<OptionHoldingWithData[]>([]);
  const [optionsLoading, setOptionsLoading] = useState(false);
  const [optionsError, setOptionsError] = useState<string | null>(null);

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
    lastPricesFetched,
    refresh, 
    addHolding, 
    removeHolding,
    updateHolding 
  } = usePortfolio({
    chartPeriod,
    onHistoryUpdate: handleHistoryUpdate
  });

  // Load settings from portfolio on first load
  useEffect(() => {
    if (portfolio && !settingsInitialized) {
      // Initialize chart period from saved setting (use default if null)
      const savedPeriod = portfolio.chart_period as ChartPeriod | null;
      if (savedPeriod) {
        setChartPeriod(savedPeriod);
      }
      // Initialize sort option from saved settings (use defaults if null)
      const savedField = portfolio.sort_field as SortOption['field'] | null;
      const savedDirection = portfolio.sort_direction as SortOption['direction'] | null;
      if (savedField || savedDirection) {
        setSortOption({
          field: savedField || 'allocation',
          direction: savedDirection || 'desc'
        });
      }
      setSettingsInitialized(true);
    }
  }, [portfolio, settingsInitialized]);

  // Save chart period when it changes (after initial load)
  const handleChartPeriodChange = useCallback((period: ChartPeriod) => {
    setChartPeriod(period);
    if (settingsInitialized) {
      api.updatePortfolio({ chart_period: period }).catch(console.error);
    }
  }, [settingsInitialized]);

  // Save sort option when it changes (after initial load)
  const handleSortOptionChange = useCallback((option: SortOption) => {
    setSortOption(option);
    if (settingsInitialized) {
      api.updatePortfolio({ 
        sort_field: option.field, 
        sort_direction: option.direction 
      }).catch(console.error);
    }
  }, [settingsInitialized]);
  
  // Track tickers to detect changes
  const tickersKey = useMemo(() => {
    return portfolio?.holdings.map(h => h.ticker).sort().join(',') ?? '';
  }, [portfolio?.holdings]);
  
  const prevPeriodRef = useRef(chartPeriod);

  // Track period changes - DON'T clear chart data immediately
  // Keep showing old data until new data arrives for smoother transitions
  useEffect(() => {
    prevPeriodRef.current = chartPeriod;
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

  // Helper to get action score for a holding (uses cached analysis from ActionScoreBadge)
  const getActionScore = useCallback((ticker: string): number | null => {
    return getCachedAnalysisScore(ticker);
  }, []);

  // Sort holdings based on current sort option
  const sortedHoldings = useMemo(() => {
    if (!portfolio?.holdings) return [];
    
    const holdings = [...portfolio.holdings];
    const { field, direction } = sortOption;
    const multiplier = direction === 'asc' ? 1 : -1;

    // Pre-compute values that require expensive calculations to ensure stable sorting
    // (calling functions during sort comparisons can cause inconsistent results)
    const periodGainCache = new Map<string, number | null>();
    const actionScoreCache = new Map<string, number | null>();
    
    for (const h of holdings) {
      periodGainCache.set(h.ticker, getPeriodGain(h.ticker));
      actionScoreCache.set(h.ticker, getActionScore(h.ticker));
    }

    holdings.sort((a, b) => {
      let comparison = 0;

      switch (field) {
        case 'ticker':
          comparison = a.ticker.localeCompare(b.ticker);
          break;
        case 'top_movers': {
          // Use absolute value - biggest movers regardless of direction
          const gainA = periodGainCache.get(a.ticker) ?? null;
          const gainB = periodGainCache.get(b.ticker) ?? null;
          // Put null values at the end (return directly to bypass multiplier)
          if (gainA === null && gainB === null) return 0;
          if (gainA === null) return 1;
          if (gainB === null) return -1;
          comparison = Math.abs(gainA) - Math.abs(gainB);
          break;
        }
        case 'period_change': {
          // Actual value - gainers vs losers
          const gainA = periodGainCache.get(a.ticker) ?? null;
          const gainB = periodGainCache.get(b.ticker) ?? null;
          // Put null values at the end (return directly to bypass multiplier)
          if (gainA === null && gainB === null) return 0;
          if (gainA === null) return 1;
          if (gainB === null) return -1;
          comparison = gainA - gainB;
          break;
        }
        case 'allocation':
          comparison = (a.allocation_pct ?? 0) - (b.allocation_pct ?? 0);
          break;
        case 'equity':
          comparison = (a.market_value ?? 0) - (b.market_value ?? 0);
          break;
        case 'gain_pct':
          comparison = (a.gain_loss_pct ?? 0) - (b.gain_loss_pct ?? 0);
          break;
        case 'gain_value':
          comparison = (a.gain_loss ?? 0) - (b.gain_loss ?? 0);
          break;
        case 'confidence': {
          // Sort by action score (0-100)
          const scoreA = actionScoreCache.get(a.ticker) ?? null;
          const scoreB = actionScoreCache.get(b.ticker) ?? null;
          // Put null values at the end (return directly to bypass multiplier)
          if (scoreA === null && scoreB === null) return 0;
          if (scoreA === null) return 1;
          if (scoreB === null) return -1;
          comparison = scoreA - scoreB;
          break;
        }
        default:
          comparison = 0;
      }

      return comparison * multiplier;
    });

    return holdings;
  }, [portfolio?.holdings, sortOption, getPeriodGain, getActionScore]);

  const handleAddHolding = async (ticker: string, shares: number, avgCost?: number) => {
    await addHolding(ticker, shares, avgCost);
  };

  const handleUpdatePosition = async (holdingId: number, data: { shares?: number; avg_cost?: number }) => {
    await updateHolding(holdingId, data);
  };

  const handleRefreshHistory = async (ticker: string) => {
    // Clear cached history on backend
    await api.clearStockHistory(ticker);
    // Clear local chart data to trigger refetch
    setHoldingChartData(prev => {
      const newData = { ...prev };
      delete newData[ticker];
      return newData;
    });
  };

  // ============================================================================
  // Options Management
  // ============================================================================

  // Fetch option holdings
  const fetchOptionHoldings = useCallback(async () => {
    setOptionsLoading(true);
    setOptionsError(null);
    try {
      const options = await api.getOptionHoldings();
      setOptionHoldings(options);
    } catch (err) {
      console.error('Failed to fetch option holdings:', err);
      setOptionsError(err instanceof Error ? err.message : 'Failed to load options');
    } finally {
      setOptionsLoading(false);
    }
  }, []);

  // Load options on mount and periodically refresh
  useEffect(() => {
    fetchOptionHoldings();
    // Refresh options every 60 seconds
    const interval = setInterval(fetchOptionHoldings, 60000);
    return () => clearInterval(interval);
  }, [fetchOptionHoldings]);

  const handleAddOption = async (option: OptionHoldingCreate) => {
    await api.addOptionHolding(option);
    await fetchOptionHoldings();
  };

  const handleDeleteOption = async (optionId: number) => {
    await api.deleteOptionHolding(optionId);
    setOptionHoldings(prev => prev.filter(o => o.id !== optionId));
  };

  // Calculate total options value
  const totalOptionsValue = useMemo(() => {
    return optionHoldings.reduce((sum, opt) => sum + (opt.position_value ?? 0), 0);
  }, [optionHoldings]);

  const totalOptionsGainLoss = useMemo(() => {
    return optionHoldings.reduce((sum, opt) => sum + (opt.gain_loss ?? 0), 0);
  }, [optionHoldings]);

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
    <ErrorBoundary>
      <div className="min-h-screen p-6 md:p-8">
        <div className="animated-bg" />
        
        <div className="max-w-7xl mx-auto">
          <Header 
            totalValue={portfolio?.total_market_value ?? 0}
            lastUpdated={lastFetched ?? undefined}
            isDataReady={allDataLoaded}
            holdings={portfolio?.holdings}
          />

          {portfolio && (
            <>
              <SectionErrorBoundary sectionName="portfolio summary">
                <PortfolioSummary portfolio={portfolio} />
              </SectionErrorBoundary>

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
                  
                  <div className="flex items-center gap-2">
                    {portfolio.holdings.length >= 2 && (
                      <button 
                        onClick={() => setShowCompareModal(true)}
                        className="btn-secondary flex items-center gap-2"
                      >
                        <Layers className="w-5 h-5" />
                        Compare
                      </button>
                    )}
                    <button 
                      onClick={() => setShowAddModal(true)}
                      className="btn-primary flex items-center gap-2"
                    >
                      <Plus className="w-5 h-5" />
                      Add Stock
                    </button>
                  </div>
                </div>

                {/* Chart Period & Sort Selectors */}
                {portfolio.holdings.length > 0 && (
                  <div className="mb-4 flex flex-wrap items-center gap-x-6 gap-y-3">
                    <div className="flex items-center gap-3">
                      <span className="text-white/50 text-sm">Chart period:</span>
                      <ChartPeriodSelector 
                        selected={chartPeriod} 
                        onSelect={handleChartPeriodChange} 
                      />
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-white/50 text-sm">Sort by:</span>
                      <SortSelector
                        value={sortOption}
                        onChange={handleSortOptionChange}
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
                          <SectionErrorBoundary sectionName={`${holding.ticker} card`}>
                            <HoldingCard
                              holding={holding}
                              history={chartData?.history ?? []}
                              referenceClose={chartData?.referenceClose ?? null}
                              isDataComplete={chartData?.isComplete ?? false}
                              expectedStart={chartData?.expectedStart ?? null}
                              actualStart={chartData?.actualStart ?? null}
                              onDelete={removeHolding}
                              onSelect={setSelectedTicker}
                              onAnalyze={setAnalyzeTicker}
                              onUpdatePosition={handleUpdatePosition}
                              onRefreshHistory={handleRefreshHistory}
                              isRefreshing={false}
                              isHistoryLoading={!chartData}
                              lastPricesFetched={lastPricesFetched}
                            />
                          </SectionErrorBoundary>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Options Section */}
              <div className="fade-in mt-8" style={{ animationDelay: '0.4s' }}>
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
                  <div className="flex items-center gap-3">
                    <TrendingUp className="w-5 h-5 text-accent-purple" />
                    <h2 className="text-xl font-semibold text-white">Options</h2>
                    <span className="text-white/40 text-sm">
                      ({optionHoldings.length} positions)
                    </span>
                    {optionHoldings.length > 0 && (
                      <>
                        <span className="text-white/30">•</span>
                        <span className="text-white/60 text-sm">
                          ${totalOptionsValue.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                        </span>
                        <span className={`text-sm font-medium ${totalOptionsGainLoss >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          ({totalOptionsGainLoss >= 0 ? '+' : ''}${totalOptionsGainLoss.toLocaleString('en-US', { minimumFractionDigits: 2 })})
                        </span>
                      </>
                    )}
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <button
                      onClick={fetchOptionHoldings}
                      disabled={optionsLoading}
                      className="btn-secondary flex items-center gap-2"
                      title="Refresh option prices"
                    >
                      <RefreshCw className={`w-4 h-4 ${optionsLoading ? 'animate-spin' : ''}`} />
                    </button>
                    <button 
                      onClick={() => setShowAddOptionModal(true)}
                      className="btn-primary flex items-center gap-2"
                    >
                      <Plus className="w-5 h-5" />
                      Add Option
                    </button>
                  </div>
                </div>

                {optionsError && (
                  <div className="glass-card p-4 mb-4 border-red-500/30">
                    <p className="text-red-400 text-sm">{optionsError}</p>
                  </div>
                )}

                {optionHoldings.length === 0 ? (
                  <div className="glass-card p-12 text-center">
                    <TrendingUp className="w-16 h-16 text-white/20 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-white mb-2">No options yet</h3>
                    <p className="text-white/50 mb-6">
                      Track your calls, puts, covered calls, and cash-secured puts
                    </p>
                    <button 
                      onClick={() => setShowAddOptionModal(true)}
                      className="btn-primary inline-flex items-center gap-2"
                    >
                      <Plus className="w-5 h-5" />
                      Add Your First Option
                    </button>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {optionHoldings.map((option, index) => (
                      <div 
                        key={option.id} 
                        className="fade-in"
                        style={{ animationDelay: `${0.1 * (index + 1)}s` }}
                      >
                        <SectionErrorBoundary sectionName={`${option.underlying_ticker} option`}>
                          <OptionHoldingCard
                            option={option}
                            onDelete={handleDeleteOption}
                          />
                        </SectionErrorBoundary>
                      </div>
                    ))}
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
        />

        <AddOptionModal
          isOpen={showAddOptionModal}
          onClose={() => setShowAddOptionModal(false)}
          onAdd={handleAddOption}
        />

        <StockDetailModal
          ticker={selectedTicker}
          onClose={() => setSelectedTicker(null)}
          initialChartPeriod={chartPeriod}
          lastPricesFetched={lastPricesFetched}
          holding={selectedTicker ? portfolio?.holdings.find(h => h.ticker === selectedTicker) : null}
        />

        {/* Stock Analysis Modal - uses shared cache with ActionScoreBadge */}
        {analyzeTicker && (
          <StockAnalysisModal
            ticker={analyzeTicker}
            onClose={() => setAnalyzeTicker(null)}
            currentPrice={portfolio?.holdings.find(h => h.ticker === analyzeTicker)?.current_price}
          />
        )}

        {/* Stock Compare Modal - overlay multiple holdings on one chart */}
        <StockCompareModal
          isOpen={showCompareModal}
          onClose={() => setShowCompareModal(false)}
          holdings={portfolio?.holdings ?? []}
          holdingChartData={holdingChartData}
          initialChartPeriod={chartPeriod}
        />
      </div>
    </ErrorBoundary>
  );
}

export default App;
