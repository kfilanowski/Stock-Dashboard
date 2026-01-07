import { useState, useMemo, useCallback, useEffect } from 'react';
import { X, Check, Layers } from 'lucide-react';
import { 
  Line, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  CartesianGrid, ComposedChart, Legend, ReferenceLine 
} from 'recharts';
import type { Holding, HistoryPoint } from '../types';
import { ChartPeriodSelector, type ChartPeriod } from './ChartPeriodSelector';
import type { HoldingChartData } from '../hooks/usePortfolio';
import * as api from '../services/api';

// Color palette for multiple stocks (extended for more selections)
const COMPARE_COLORS = [
  '#22c55e', // green
  '#3b82f6', // blue  
  '#f59e0b', // amber
  '#ef4444', // red
  '#a855f7', // purple
  '#06b6d4', // cyan
  '#ec4899', // pink
  '#14b8a6', // teal
  '#84cc16', // lime
  '#f97316', // orange
  '#8b5cf6', // violet
  '#0ea5e9', // sky
  '#d946ef', // fuchsia
  '#10b981', // emerald
  '#eab308', // yellow
  '#64748b', // slate
];

// Extended hours trading window (4 AM to 8 PM Eastern)
const EXTENDED_START_MINUTES = 4 * 60;    // 4:00 AM = 240 minutes
const EXTENDED_END_MINUTES = 20 * 60;     // 8:00 PM = 1200 minutes

// Parse time string "HH:MM" to minutes since midnight
function parseTimeToMinutes(timeStr: string): number {
  const [hours, minutes] = timeStr.split(':').map(Number);
  if (isNaN(hours) || isNaN(minutes)) return -1;
  return hours * 60 + minutes;
}

// Check if this is intraday data (has time component)
function isIntradayData(history: HistoryPoint[]): boolean {
  if (!history.length) return false;
  return history[0].date.includes(' ');
}

// Convert date string to minutes since midnight (for X-axis positioning)
function dateToMinutes(dateStr: string): number {
  if (!dateStr.includes(' ')) return 0;
  const timePart = dateStr.split(' ')[1];
  return parseTimeToMinutes(timePart);
}

// Get the trading day from a date string
function getTradingDay(dateStr: string): string {
  if (dateStr.includes(' ')) {
    return dateStr.split(' ')[0];
  }
  return dateStr;
}

// Convert xValue (minutes since start of chart) to time string for X-axis labels
function minutesToTimeLabel(
  minutes: number, 
  tradingDays?: string[],
  showDayNameOnly?: boolean
): string {
  const minutesPerDay = EXTENDED_END_MINUTES - EXTENDED_START_MINUTES;
  const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  
  let dayIndex: number;
  let minuteWithinDay: number;
  
  if (minutes > 0 && minutes % minutesPerDay === 0) {
    dayIndex = (minutes / minutesPerDay) - 1;
    minuteWithinDay = minutesPerDay;
  } else {
    dayIndex = Math.floor(minutes / minutesPerDay);
    minuteWithinDay = minutes % minutesPerDay;
  }
  
  if (showDayNameOnly && tradingDays && tradingDays[dayIndex]) {
    const dateStr = tradingDays[dayIndex];
    const [year, month, day] = dateStr.split('-').map(Number);
    const date = new Date(year, month - 1, day);
    return dayNames[date.getDay()];
  }
  
  const minutesSinceMidnight = minuteWithinDay + EXTENDED_START_MINUTES;
  
  const hours = Math.floor(minutesSinceMidnight / 60);
  const mins = minutesSinceMidnight % 60;
  const ampm = hours >= 12 ? 'PM' : 'AM';
  const hours12 = hours % 12 || 12;
  
  const timeStr = mins === 0 ? `${hours12}${ampm}` : `${hours12}:${mins.toString().padStart(2, '0')}${ampm}`;
  
  return timeStr;
}

type ViewMode = 'price' | 'percent';

interface StockCompareModalProps {
  isOpen: boolean;
  onClose: () => void;
  holdings: Holding[];
  holdingChartData: Record<string, HoldingChartData>;
  /** Initial chart period when modal opens */
  initialChartPeriod?: ChartPeriod;
}

interface ProcessedDataPoint {
  date: string;
  xValue: number;
  [key: string]: number | string; // Dynamic keys for each ticker's price/percent
}

export function StockCompareModal({
  isOpen,
  onClose,
  holdings,
  holdingChartData,
  initialChartPeriod = '1d'
}: StockCompareModalProps) {
  const [selectedTickers, setSelectedTickers] = useState<Set<string>>(new Set());
  const [viewMode, setViewMode] = useState<ViewMode>('percent');
  // Local chart period state - isolated from parent dashboard
  const [chartPeriod, setChartPeriod] = useState<ChartPeriod>(initialChartPeriod);
  // Local chart data for this modal (fetched when period differs from parent)
  const [localChartData, setLocalChartData] = useState<Record<string, HoldingChartData>>({});
  const [loadingTickers, setLoadingTickers] = useState<Set<string>>(new Set());

  // Reset state when modal opens
  useEffect(() => {
    if (isOpen) {
      setChartPeriod(initialChartPeriod);
      setLocalChartData({});
      setLoadingTickers(new Set());
    }
  }, [isOpen, initialChartPeriod]);

  // Fetch history for selected tickers when period changes
  useEffect(() => {
    if (!isOpen || selectedTickers.size === 0) return;

    const fetchHistoryForTickers = async () => {
      for (const ticker of selectedTickers) {
        // Skip if already loading or already have data for this period
        if (loadingTickers.has(ticker)) continue;
        
        // Check if we need to fetch (period differs from parent data)
        const parentData = holdingChartData[ticker];
        const localData = localChartData[ticker];
        
        // If we have local data for this ticker, use it
        if (localData) continue;
        
        // Fetch new data for this period
        setLoadingTickers(prev => new Set(prev).add(ticker));
        
        try {
          const result = await api.getStockHistory(ticker, chartPeriod);
          setLocalChartData(prev => ({
            ...prev,
            [ticker]: {
              history: result.history,
              referenceClose: result.reference_close,
              isComplete: result.is_complete,
              expectedStart: result.expected_start,
              actualStart: result.actual_start
            }
          }));
        } catch (err) {
          console.error(`Failed to fetch history for ${ticker}:`, err);
        } finally {
          setLoadingTickers(prev => {
            const next = new Set(prev);
            next.delete(ticker);
            return next;
          });
        }
      }
    };

    fetchHistoryForTickers();
  }, [isOpen, selectedTickers, chartPeriod, holdingChartData, localChartData, loadingTickers]);

  // Clear local data when period changes
  const handlePeriodChange = useCallback((period: ChartPeriod) => {
    setChartPeriod(period);
    setLocalChartData({}); // Clear to trigger refetch
  }, []);

  // Get chart data for a ticker (prefer local data, fallback to parent data)
  const getChartData = useCallback((ticker: string): HoldingChartData | undefined => {
    return localChartData[ticker] || holdingChartData[ticker];
  }, [localChartData, holdingChartData]);

  // Sort holdings alphabetically
  const sortedHoldings = useMemo(() => {
    return [...holdings].sort((a, b) => a.ticker.localeCompare(b.ticker));
  }, [holdings]);

  // Toggle a ticker's selection
  const toggleTicker = useCallback((ticker: string) => {
    setSelectedTickers(prev => {
      const next = new Set(prev);
      if (next.has(ticker)) {
        next.delete(ticker);
      } else {
        next.add(ticker);
      }
      return next;
    });
  }, []);

  // Select all / clear all
  const selectAll = useCallback(() => {
    const availableTickers = sortedHoldings
      .filter(h => getChartData(h.ticker)?.history?.length || holdingChartData[h.ticker]?.history?.length)
      .map(h => h.ticker);
    setSelectedTickers(new Set(availableTickers));
  }, [sortedHoldings, getChartData, holdingChartData]);

  const clearAll = useCallback(() => {
    setSelectedTickers(new Set());
  }, []);

  // Get color for a ticker
  const getTickerColor = useCallback((ticker: string): string => {
    const tickers = Array.from(selectedTickers);
    const index = tickers.indexOf(ticker);
    return COMPARE_COLORS[index % COMPARE_COLORS.length];
  }, [selectedTickers]);

  // Check if data is intraday
  const hasIntradayData = useMemo(() => {
    for (const ticker of selectedTickers) {
      const data = getChartData(ticker);
      if (data?.history?.length && isIntradayData(data.history)) {
        return true;
      }
    }
    return false;
  }, [selectedTickers, getChartData]);

  // Get all unique trading days across selected stocks
  const tradingDays = useMemo(() => {
    const allDays = new Set<string>();
    for (const ticker of selectedTickers) {
      const data = getChartData(ticker);
      if (data?.history?.length) {
        for (const point of data.history) {
          allDays.add(getTradingDay(point.date));
        }
      }
    }
    return [...allDays].sort();
  }, [selectedTickers, getChartData]);

  // Process and merge chart data for all selected stocks
  const { processedData, yDomain, xDomain, numTradingDays } = useMemo(() => {
    if (selectedTickers.size === 0) {
      return { 
        processedData: [] as ProcessedDataPoint[], 
        yDomain: [0, 100] as [number, number],
        xDomain: [0, 100] as [number, number],
        numTradingDays: 0
      };
    }

    // Collect all dates across all selected tickers
    const allDates = new Set<string>();
    const tickerData: Record<string, { history: HistoryPoint[]; referenceClose: number | null }> = {};

    for (const ticker of selectedTickers) {
      const data = getChartData(ticker);
      if (data?.history?.length) {
        tickerData[ticker] = {
          history: data.history,
          referenceClose: data.referenceClose
        };
        for (const point of data.history) {
          allDates.add(point.date);
        }
      }
    }

    // Sort dates
    const sortedDates = [...allDates].sort();
    const isIntraday = sortedDates.length > 0 && sortedDates[0].includes(' ');

    // Calculate minutesPerDay for intraday positioning
    const minutesPerDay = EXTENDED_END_MINUTES - EXTENDED_START_MINUTES;
    const days = [...new Set(sortedDates.map(d => getTradingDay(d)))].sort();
    const numDays = days.length;

    // Build merged data points
    const dataByDate: Record<string, ProcessedDataPoint> = {};
    let minY = Infinity;
    let maxY = -Infinity;

    for (const ticker of selectedTickers) {
      const { history, referenceClose } = tickerData[ticker] || { history: [], referenceClose: null };
      
      for (const point of history) {
        const date = point.date;
        
        if (!dataByDate[date]) {
          // Calculate xValue for this date
          let xValue: number;
          if (isIntraday) {
            const dayIndex = days.indexOf(getTradingDay(date));
            const minuteOfDay = dateToMinutes(date);
            xValue = (dayIndex * minutesPerDay) + (minuteOfDay - EXTENDED_START_MINUTES);
          } else {
            xValue = sortedDates.indexOf(date);
          }
          
          dataByDate[date] = { date, xValue };
        }

        // Store price value
        const priceKey = `${ticker}_price`;
        dataByDate[date][priceKey] = point.close;

        // Store percent change value (if we have reference close)
        if (referenceClose && referenceClose > 0) {
          const percentKey = `${ticker}_percent`;
          const percentChange = ((point.close - referenceClose) / referenceClose) * 100;
          dataByDate[date][percentKey] = percentChange;

          if (viewMode === 'percent') {
            minY = Math.min(minY, percentChange);
            maxY = Math.max(maxY, percentChange);
          }
        }

        if (viewMode === 'price') {
          minY = Math.min(minY, point.close);
          maxY = Math.max(maxY, point.close);
        }
      }
    }

    // Convert to array and sort by xValue
    const processed = Object.values(dataByDate).sort((a, b) => a.xValue - b.xValue);

    // Calculate domains
    const padding = (maxY - minY) * 0.1 || 1;
    const yDomainCalc: [number, number] = [
      minY === Infinity ? 0 : minY - padding,
      maxY === -Infinity ? 100 : maxY + padding
    ];

    const xDomainCalc: [number, number] = isIntraday
      ? [0, numDays * minutesPerDay]
      : [0, Math.max(sortedDates.length - 1, 1)];

    return { 
      processedData: processed, 
      yDomain: yDomainCalc, 
      xDomain: xDomainCalc,
      numTradingDays: numDays
    };
  }, [selectedTickers, getChartData, viewMode]);

  // Generate X-axis ticks
  const xAxisTicks = useMemo(() => {
    if (!hasIntradayData || numTradingDays === 0) return undefined;
    
    const minutesPerDay = EXTENDED_END_MINUTES - EXTENDED_START_MINUTES;
    const ticks: number[] = [];
    
    if (numTradingDays === 1) {
      [4, 8, 12, 16, 20].forEach(hour => {
        ticks.push((hour * 60) - EXTENDED_START_MINUTES);
      });
    } else {
      for (let day = 0; day < numTradingDays; day++) {
        ticks.push(day * minutesPerDay + (12 * 60 - EXTENDED_START_MINUTES));
      }
    }
    
    return ticks;
  }, [hasIntradayData, numTradingDays]);

  // Format X-axis labels
  const formatXAxis = useCallback((value: number | string) => {
    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    
    if (chartPeriod === '1d' && hasIntradayData && typeof value === 'number') {
      return minutesToTimeLabel(value, tradingDays, false);
    }
    
    if (['3d', '1w'].includes(chartPeriod) && hasIntradayData && typeof value === 'number') {
      return minutesToTimeLabel(value, tradingDays, true);
    }
    
    // For daily data, find the date at this index
    if (typeof value === 'number' && processedData[value]) {
      const dateStr = processedData[value].date.split(' ')[0];
      const [year, month, day] = dateStr.split('-').map(Number);
      
      if (['1mo', '3mo'].includes(chartPeriod)) {
        return `${day}`;
      }
      if (['6mo', 'ytd', '1y'].includes(chartPeriod)) {
        return monthNames[month - 1];
      }
      if (chartPeriod === '2y') {
        return `${month}`;
      }
      if (chartPeriod === '5y') {
        return `${year}`;
      }
      return `${monthNames[month - 1]} ${day}`;
    }
    
    return '';
  }, [chartPeriod, hasIntradayData, tradingDays, processedData]);

  // Format Y-axis labels
  const formatYAxis = useCallback((value: number) => {
    if (viewMode === 'percent') {
      return `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`;
    }
    return `$${value.toFixed(0)}`;
  }, [viewMode]);

  // Custom tooltip
  const CustomTooltip = useCallback(({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;
    
    const dataPoint = payload[0]?.payload;
    const dateStr = dataPoint?.date ?? '';
    
    // Format date
    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    let formattedDate = dateStr;
    if (dateStr.includes(' ')) {
      const [datePart, timePart] = dateStr.split(' ');
      const [year, month, day] = datePart.split('-').map(Number);
      const [hours, mins] = timePart.split(':').map(Number);
      const ampm = hours >= 12 ? 'PM' : 'AM';
      const hours12 = hours % 12 || 12;
      formattedDate = `${monthNames[month - 1]} ${day}, ${hours12}:${mins.toString().padStart(2, '0')} ${ampm}`;
    } else if (dateStr.includes('-')) {
      const [year, month, day] = dateStr.split('-').map(Number);
      formattedDate = `${monthNames[month - 1]} ${day}, ${year}`;
    }
    
    return (
      <div className="bg-[rgba(10,10,15,0.95)] border border-white/10 rounded-lg px-3 py-2 shadow-xl min-w-[140px]">
        <p className="text-white/60 text-xs mb-2">{formattedDate}</p>
        {payload.map((entry: any, idx: number) => {
          const ticker = entry.dataKey.replace('_price', '').replace('_percent', '');
          const isPercent = entry.dataKey.endsWith('_percent');
          const value = entry.value;
          
          return (
            <div key={idx} className="flex items-center justify-between gap-3 text-sm">
              <span className="flex items-center gap-1.5">
                <span 
                  className="w-2 h-2 rounded-full" 
                  style={{ backgroundColor: entry.color }}
                />
                <span className="text-white/80">{ticker}</span>
              </span>
              <span className={`font-medium ${
                isPercent 
                  ? value >= 0 ? 'text-green-400' : 'text-red-400'
                  : 'text-white'
              }`}>
                {isPercent 
                  ? `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
                  : `$${value.toFixed(2)}`
                }
              </span>
            </div>
          );
        })}
      </div>
    );
  }, []);

  // Calculate period gains for selected stocks
  const periodGains = useMemo(() => {
    const gains: Record<string, number | null> = {};
    for (const ticker of selectedTickers) {
      const data = getChartData(ticker);
      if (data?.history?.length && data.referenceClose && data.referenceClose > 0) {
        const lastClose = data.history[data.history.length - 1].close;
        gains[ticker] = ((lastClose - data.referenceClose) / data.referenceClose) * 100;
      } else {
        gains[ticker] = null;
      }
    }
    return gains;
  }, [selectedTickers, getChartData]);

  if (!isOpen) return null;

  const selectedArray = Array.from(selectedTickers);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal */}
      <div className="glass-card p-6 w-full max-w-5xl max-h-[90vh] overflow-y-auto relative z-10 fade-in">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-accent-cyan/30 to-accent-purple/30 flex items-center justify-center border border-white/10">
              <Layers className="w-6 h-6 text-accent-cyan" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">Compare Holdings</h2>
              <p className="text-white/50 text-sm">
                Select stocks to overlay their price charts
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/10 transition-colors"
          >
            <X className="w-6 h-6 text-white/70" />
          </button>
        </div>

        <div className="flex flex-col lg:flex-row gap-6">
          {/* Stock Selector Panel */}
          <div className="lg:w-64 shrink-0">
            <div className="flex items-center justify-between mb-3">
              <span className="text-white/70 text-sm font-medium">Select Stocks</span>
              <div className="flex gap-2">
                <button 
                  onClick={selectAll}
                  className="text-xs text-accent-cyan hover:text-accent-cyan/80 transition-colors"
                >
                  All
                </button>
                <span className="text-white/30">|</span>
                <button 
                  onClick={clearAll}
                  className="text-xs text-white/50 hover:text-white/70 transition-colors"
                >
                  Clear
                </button>
              </div>
            </div>
            
            <div className="space-y-2 max-h-80 overflow-y-auto pr-2">
              {sortedHoldings.map((holding) => {
                const chartData = getChartData(holding.ticker);
                const hasData = chartData?.history?.length > 0 || holdingChartData[holding.ticker]?.history?.length > 0;
                const isSelected = selectedTickers.has(holding.ticker);
                const isLoading = loadingTickers.has(holding.ticker);
                const color = isSelected ? getTickerColor(holding.ticker) : undefined;
                const gain = periodGains[holding.ticker];
                
                return (
                  <button
                    key={holding.id}
                    onClick={() => toggleTicker(holding.ticker)}
                    className={`w-full flex items-center gap-3 p-3 rounded-lg border transition-all ${
                      isSelected
                        ? 'bg-white/10 border-white/20'
                        : 'bg-white/5 border-white/10 hover:bg-white/10'
                    }`}
                  >
                    {/* Checkbox */}
                    <div 
                      className={`w-5 h-5 rounded flex items-center justify-center border transition-all ${
                        isSelected 
                          ? 'border-transparent' 
                          : 'border-white/30'
                      }`}
                      style={{ backgroundColor: isSelected ? color : 'transparent' }}
                    >
                      {isSelected && <Check className="w-3.5 h-3.5 text-white" />}
                    </div>
                    
                    {/* Ticker info */}
                    <div className="flex-1 text-left">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-white">{holding.ticker}</span>
                        {isLoading && (
                          <span className="text-xs text-accent-cyan/70">Loading...</span>
                        )}
                      </div>
                      <div className="flex items-center gap-2 text-xs">
                        <span className="text-white/50">
                          ${holding.current_price?.toFixed(2) ?? '—'}
                        </span>
                        {isSelected && gain !== null && (
                          <span className={gain >= 0 ? 'text-green-400' : 'text-red-400'}>
                            {gain >= 0 ? '+' : ''}{gain.toFixed(2)}%
                          </span>
                        )}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Chart Area */}
          <div className="flex-1 min-w-0">
            {/* Controls */}
            <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
              {/* Period Selector */}
              <ChartPeriodSelector 
                selected={chartPeriod} 
                onSelect={handlePeriodChange} 
              />
              
              {/* View Mode Toggle */}
              <div className="flex items-center gap-1 bg-white/5 rounded-lg p-1 border border-white/10">
                <button
                  onClick={() => setViewMode('percent')}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    viewMode === 'percent'
                      ? 'bg-gradient-to-r from-accent-cyan to-accent-purple text-white'
                      : 'text-white/60 hover:text-white/80'
                  }`}
                >
                  % Change
                </button>
                <button
                  onClick={() => setViewMode('price')}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    viewMode === 'price'
                      ? 'bg-gradient-to-r from-accent-cyan to-accent-purple text-white'
                      : 'text-white/60 hover:text-white/80'
                  }`}
                >
                  Price
                </button>
              </div>
            </div>

            {/* Chart */}
            <div className="h-80 bg-white/5 rounded-xl p-4 border border-white/10">
              {selectedTickers.size === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-white/50">
                  <Layers className="w-12 h-12 mb-3 opacity-50" />
                  <p className="text-sm">Select at least 2 stocks to compare</p>
                </div>
              ) : selectedTickers.size === 1 ? (
                <div className="h-full flex flex-col items-center justify-center text-white/50">
                  <p className="text-sm">Select one more stock to compare</p>
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={processedData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    
                    {hasIntradayData ? (
                      <XAxis 
                        dataKey="xValue"
                        type="number"
                        domain={xDomain}
                        stroke="rgba(255,255,255,0.3)"
                        tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 11 }}
                        tickFormatter={formatXAxis}
                        ticks={xAxisTicks}
                      />
                    ) : (
                      <XAxis 
                        dataKey="xValue"
                        type="number"
                        domain={xDomain}
                        stroke="rgba(255,255,255,0.3)"
                        tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 11 }}
                        tickFormatter={formatXAxis}
                      />
                    )}
                    
                    <YAxis 
                      stroke="rgba(255,255,255,0.3)"
                      tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                      domain={yDomain}
                      tickFormatter={formatYAxis}
                    />
                    
                    {/* Zero reference line for percent mode */}
                    {viewMode === 'percent' && (
                      <ReferenceLine 
                        y={0} 
                        stroke="rgba(255,255,255,0.3)" 
                        strokeDasharray="4 4"
                        strokeWidth={1}
                      />
                    )}
                    
                    <Tooltip content={<CustomTooltip />} />
                    
                    <Legend 
                      verticalAlign="top"
                      height={36}
                      formatter={(value: string) => {
                        const ticker = value.replace('_price', '').replace('_percent', '');
                        return <span className="text-white/80 text-xs">{ticker}</span>;
                      }}
                    />
                    
                    {/* Render a line for each selected ticker */}
                    {selectedArray.map((ticker) => {
                      const color = getTickerColor(ticker);
                      const dataKey = viewMode === 'percent' 
                        ? `${ticker}_percent` 
                        : `${ticker}_price`;
                      
                      return (
                        <Line
                          key={ticker}
                          type="monotone"
                          dataKey={dataKey}
                          name={ticker}
                          stroke={color}
                          strokeWidth={2}
                          dot={false}
                          activeDot={{ r: 5, fill: color, stroke: '#0a0a0f', strokeWidth: 2 }}
                          connectNulls
                          isAnimationActive={false}
                        />
                      );
                    })}
                  </ComposedChart>
                </ResponsiveContainer>
              )}
            </div>

            {/* Period Gain Summary */}
            {selectedTickers.size >= 2 && (
              <div className="mt-4 flex flex-wrap gap-4">
                {selectedArray.map(ticker => {
                  const gain = periodGains[ticker];
                  const color = getTickerColor(ticker);
                  
                  return (
                    <div key={ticker} className="flex items-center gap-2">
                      <span 
                        className="w-3 h-3 rounded-full" 
                        style={{ backgroundColor: color }}
                      />
                      <span className="text-white/70 text-sm">{ticker}</span>
                      {gain !== null ? (
                        <span className={`text-sm font-medium ${
                          gain >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {gain >= 0 ? '+' : ''}{gain.toFixed(2)}%
                        </span>
                      ) : (
                        <span className="text-white/40 text-sm">—</span>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

