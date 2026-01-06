import { useEffect, useState, useMemo, useRef } from 'react';
import { X, TrendingUp, TrendingDown, Calendar, BarChart2 } from 'lucide-react';
import { 
  Line, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  CartesianGrid, ReferenceLine, Area, ComposedChart, ReferenceArea 
} from 'recharts';
import type { StockData, HistoryPoint } from '../types';
import * as api from '../services/api';
import { ChartPeriodSelector, type ChartPeriod } from './ChartPeriodSelector';

// Extended hours trading window (4 AM to 8 PM Eastern)
const EXTENDED_START_MINUTES = 4 * 60;    // 4:00 AM = 240 minutes
const EXTENDED_END_MINUTES = 20 * 60;     // 8:00 PM = 1200 minutes

// Market hours in Eastern Time (24-hour format)
const MARKET_OPEN = 9 * 60 + 30;  // 9:30 AM = 570 minutes
const MARKET_CLOSE = 16 * 60;     // 4:00 PM = 960 minutes

// Parse time string "HH:MM" to minutes since midnight
function parseTimeToMinutes(timeStr: string): number {
  const [hours, minutes] = timeStr.split(':').map(Number);
  if (isNaN(hours) || isNaN(minutes)) return -1;
  return hours * 60 + minutes;
}

// Check if a time is during regular market hours
function isMarketHours(dateStr: string): boolean {
  if (!dateStr.includes(' ')) return true; // Daily data, assume market hours
  const timePart = dateStr.split(' ')[1];
  const minutes = parseTimeToMinutes(timePart);
  return minutes >= MARKET_OPEN && minutes < MARKET_CLOSE;
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
function minutesToTimeLabel(minutes: number, numDays: number): string {
  const minutesPerDay = EXTENDED_END_MINUTES - EXTENDED_START_MINUTES; // 960 (4AM to 8PM)
  
  // Calculate which day and minute within day
  // For values at exact day boundaries (960, 1920, etc), treat as end of previous day
  let dayIndex: number;
  let minuteWithinDay: number;
  
  if (minutes > 0 && minutes % minutesPerDay === 0) {
    // Exact end of a day (8PM) - belongs to the previous day
    dayIndex = (minutes / minutesPerDay) - 1;
    minuteWithinDay = minutesPerDay; // 960 = 8PM position
  } else {
    dayIndex = Math.floor(minutes / minutesPerDay);
    minuteWithinDay = minutes % minutesPerDay;
  }
  
  // Convert to actual clock time (minuteWithinDay is 0-960, representing 4AM-8PM)
  // 0 = 4AM (240 min from midnight), 960 = 8PM (1200 min from midnight)
  const minutesSinceMidnight = minuteWithinDay + EXTENDED_START_MINUTES;
  
  const hours = Math.floor(minutesSinceMidnight / 60);
  const mins = minutesSinceMidnight % 60;
  const ampm = hours >= 12 ? 'PM' : 'AM';
  const hours12 = hours % 12 || 12;
  
  // For clean hour marks, don't show minutes
  const timeStr = mins === 0 ? `${hours12}${ampm}` : `${hours12}:${mins.toString().padStart(2, '0')}${ampm}`;
  
  if (numDays > 1) {
    return `D${dayIndex + 1} ${timeStr}`;
  }
  return timeStr;
}

interface StockDetailModalProps {
  ticker: string | null;
  onClose: () => void;
  chartPeriod: ChartPeriod;
  onChartPeriodChange: (period: ChartPeriod) => void;
  lastPricesFetched?: Date | null;
}

export function StockDetailModal({ ticker, onClose, chartPeriod, onChartPeriodChange, lastPricesFetched }: StockDetailModalProps) {
  const [stock, setStock] = useState<StockData | null>(null);
  const [loading, setLoading] = useState(false);
  const [chartHistory, setChartHistory] = useState<HistoryPoint[]>([]);
  const [referenceClose, setReferenceClose] = useState<number | null>(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [isDataComplete, setIsDataComplete] = useState(true);
  const [expectedStart, setExpectedStart] = useState<string | null>(null);
  const [actualStart, setActualStart] = useState<string | null>(null);
  
  // Ping animation state - triggers when prices API returns
  const [pingKey, setPingKey] = useState(0);
  const [showPing, setShowPing] = useState(false);
  const prevTimestampRef = useRef<number | null>(null);
  const isFirstPricesFetch = useRef(true);

  // Fetch stock data on mount
  useEffect(() => {
    if (!ticker) {
      setStock(null);
      return;
    }

    const fetchStock = async () => {
      setLoading(true);
      try {
        const data = await api.getStock(ticker);
        setStock(data);
      } catch (err) {
        console.error('Failed to fetch stock:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchStock();
  }, [ticker]);

  // Fetch chart history when period changes
  useEffect(() => {
    if (!ticker) return;

    // Clear old data immediately when period changes to avoid stale display
    setChartHistory([]);
    setReferenceClose(null);

    const fetchHistory = async () => {
      setChartLoading(true);
      try {
        const result = await api.getStockHistory(ticker, chartPeriod);
        setChartHistory(result.history);
        setReferenceClose(result.reference_close);
        setIsDataComplete(result.is_complete);
        setExpectedStart(result.expected_start);
        setActualStart(result.actual_start);
      } catch (err) {
        console.error('Failed to fetch history:', err);
        setChartHistory([]);
        setReferenceClose(null);
        setIsDataComplete(false);
        setExpectedStart(null);
        setActualStart(null);
      } finally {
        setChartLoading(false);
      }
    };

    fetchHistory();
  }, [ticker, chartPeriod]);

  // Trigger ping when prices API returns (not just when data changes)
  useEffect(() => {
    const currentTimestamp = lastPricesFetched?.getTime() ?? null;
    
    // Skip the first render
    if (isFirstPricesFetch.current) {
      isFirstPricesFetch.current = false;
      prevTimestampRef.current = currentTimestamp;
      return;
    }
    
    // Trigger ping when prices API returns with data
    if (currentTimestamp !== null && currentTimestamp !== prevTimestampRef.current) {
      setPingKey(prev => prev + 1);
      setShowPing(true);
      const timer = setTimeout(() => setShowPing(false), 1200);
      prevTimestampRef.current = currentTimestamp;
      return () => clearTimeout(timer);
    }
    
    prevTimestampRef.current = currentTimestamp;
  }, [lastPricesFetched]);

  // Check if this is an intraday period
  const isIntraday = ['1d', '3d', '1w'].includes(chartPeriod);
  const hasIntradayData = useMemo(() => isIntradayData(chartHistory), [chartHistory]);

  // Calculate period gain
  const periodGain = useMemo(() => {
    if (!chartHistory.length || referenceClose === null || referenceClose === 0) return null;
    const latestClose = chartHistory[chartHistory.length - 1]?.close ?? 0;
    return ((latestClose - referenceClose) / referenceClose) * 100;
  }, [chartHistory, referenceClose]);

  // Determine if current price is above or below reference
  const isAboveReference = useMemo(() => {
    if (periodGain === null) return true;
    return periodGain >= 0;
  }, [periodGain]);

  // Line and gradient colors based on performance
  const lineColor = isAboveReference ? '#22c55e' : '#ef4444'; // green-500 or red-500
  const gradientId = `colorClose-${isAboveReference ? 'green' : 'red'}`;

  // Calculate Y-axis domain to include reference close
  const yAxisDomain = useMemo(() => {
    if (!chartHistory.length) return ['auto', 'auto'] as const;
    
    const closes = chartHistory.map(h => h.close).filter(c => c > 0);
    if (!closes.length) return ['auto', 'auto'] as const;
    
    let min = Math.min(...closes);
    let max = Math.max(...closes);
    
    // Include reference close in the domain
    if (referenceClose !== null && referenceClose > 0) {
      min = Math.min(min, referenceClose);
      max = Math.max(max, referenceClose);
    }
    
    // Include SMA if visible
    if (!isIntraday && stock?.sma_200) {
      min = Math.min(min, stock.sma_200);
      max = Math.max(max, stock.sma_200);
    }
    
    // Add 5% padding
    const padding = (max - min) * 0.05;
    return [min - padding, max + padding] as [number, number];
  }, [chartHistory, referenceClose, isIntraday, stock?.sma_200]);

  // Process chart data with time-based positioning for intraday
  const { processedHistory, extendedHoursRanges, xDomain, numTradingDays } = useMemo(() => {
    if (!chartHistory.length) {
      return { 
        processedHistory: [], 
        extendedHoursRanges: [], 
        xDomain: [0, 100] as [number, number],
        numTradingDays: 1
      };
    }
    
    if (!hasIntradayData) {
      // For daily data, use simple index-based positioning
      const processed = chartHistory.map((h, idx) => ({
        ...h,
        xValue: idx,
        isExtendedHours: false
      }));
      // Ensure domain has width even with single data point
      const maxIdx = Math.max(chartHistory.length - 1, 1);
      return { 
        processedHistory: processed, 
        extendedHoursRanges: [], 
        xDomain: [0, maxIdx] as [number, number],
        numTradingDays: 0
      };
    }
    
    // For intraday data, use time-based positioning
    const tradingDays = [...new Set(chartHistory.map(h => getTradingDay(h.date)))].sort();
    const numDays = tradingDays.length;
    const minutesPerDay = EXTENDED_END_MINUTES - EXTENDED_START_MINUTES; // 960 minutes (4 AM to 8 PM)
    
    const processed = chartHistory.map((h) => {
      const dayIndex = tradingDays.indexOf(getTradingDay(h.date));
      const minuteOfDay = dateToMinutes(h.date);
      // Position within the full time range: dayIndex * dayWidth + position within day
      const xValue = (dayIndex * minutesPerDay) + (minuteOfDay - EXTENDED_START_MINUTES);
      
      return {
        ...h,
        xValue,
        minuteOfDay,
        isExtendedHours: !isMarketHours(h.date)
      };
    });
    
    // FIXED domain: Always show full trading day(s) from 4 AM to 8 PM
    const domain: [number, number] = [0, numDays * minutesPerDay];
    
    // Find extended hours ranges (for shading) using xValue
    const ranges: { start: number; end: number }[] = [];
    let rangeStart: number | null = null;
    
    for (let i = 0; i < processed.length; i++) {
      if (processed[i].isExtendedHours) {
        if (rangeStart === null) rangeStart = processed[i].xValue;
      } else {
        if (rangeStart !== null) {
          ranges.push({ start: rangeStart, end: processed[i - 1].xValue });
          rangeStart = null;
        }
      }
    }
    if (rangeStart !== null && processed.length > 0) {
      ranges.push({ start: rangeStart, end: processed[processed.length - 1].xValue });
    }
    
    return { processedHistory: processed, extendedHoursRanges: ranges, xDomain: domain, numTradingDays: numDays };
  }, [chartHistory, hasIntradayData]);

  // Generate nice tick values for intraday X-axis
  const xAxisTicks = useMemo(() => {
    if (!hasIntradayData || numTradingDays === 0) return undefined;
    
    const minutesPerDay = EXTENDED_END_MINUTES - EXTENDED_START_MINUTES; // 960 minutes
    const ticks: number[] = [];
    
    if (numTradingDays === 1) {
      // 1 day: Show 4AM, 8AM, 12PM, 4PM, 8PM (5 ticks)
      [4, 8, 12, 16, 20].forEach(hour => {
        ticks.push((hour * 60) - EXTENDED_START_MINUTES);
      });
    } else if (numTradingDays <= 3) {
      // 3 days: Show 4AM and 12PM per day only (6 ticks max)
      for (let day = 0; day < numTradingDays; day++) {
        ticks.push(day * minutesPerDay); // 4AM
        ticks.push(day * minutesPerDay + (12 * 60 - EXTENDED_START_MINUTES)); // 12PM
      }
    } else {
      // 1 week (5 days): Show only 9AM per day (market open-ish)
      for (let day = 0; day < numTradingDays; day++) {
        ticks.push(day * minutesPerDay + (9 * 60 - EXTENDED_START_MINUTES)); // 9AM
      }
    }
    
    console.log('xAxisTicks:', { numTradingDays, ticks, hasIntradayData });
    return ticks;
  }, [hasIntradayData, numTradingDays]);

  // Convert 24-hour time (HH:MM) to 12-hour format
  const formatTime12Hour = (time24: string): string => {
    const [hours, minutes] = time24.split(':').map(Number);
    if (isNaN(hours) || isNaN(minutes)) return time24;
    const ampm = hours >= 12 ? 'PM' : 'AM';
    const hours12 = hours % 12 || 12;
    return `${hours12}:${minutes.toString().padStart(2, '0')} ${ampm}`;
  };

  // Format date/time properly for tooltip
  // Parse manually to avoid timezone issues (new Date("YYYY-MM-DD") is UTC, causes day-off errors)
  const formatDate = (dateStr: string, includeTime: boolean): string => {
    if (!dateStr) return '';
    
    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    const currentYear = new Date().getFullYear();
    
    // Check if it contains time (format: "YYYY-MM-DD HH:MM")
    if (dateStr.includes(' ')) {
      const [datePart, timePart] = dateStr.split(' ');
      const [year, month, day] = datePart.split('-').map(Number);
      if (isNaN(year) || isNaN(month) || isNaN(day)) return dateStr;
      
      const time12 = formatTime12Hour(timePart);
      if (includeTime) {
        // Include year if different from current year
        const formattedDate = year !== currentYear
          ? `${year} ${monthNames[month - 1]} ${day}`
          : `${monthNames[month - 1]} ${day}`;
        return `${formattedDate} ${time12}`;
      }
      return time12; // Just time for X-axis
    }
    
    // Daily format (YYYY-MM-DD)
    const [year, month, day] = dateStr.split('-').map(Number);
    if (isNaN(year) || isNaN(month) || isNaN(day)) return dateStr;
    
    if (includeTime) {
      // Create date in local time for weekday calculation
      const localDate = new Date(year, month - 1, day);
      const weekday = dayNames[localDate.getDay()];
      // Always include year for daily data in tooltip
      return `${weekday}, ${monthNames[month - 1]} ${day}, ${year}`;
    }
    // Include year if different from current year (for X-axis labels)
    return year !== currentYear 
      ? `${year} ${monthNames[month - 1]} ${day}`
      : `${monthNames[month - 1]} ${day}`;
  };

  // Format X-axis labels
  const formatXAxis = (value: number | string) => {
    if (hasIntradayData && typeof value === 'number') {
      return minutesToTimeLabel(value, numTradingDays);
    }
    // Daily data - value is the date string
    if (typeof value === 'string') {
      return formatDate(value, false);
    }
    return '';
  };

  // Calculate gain % for a specific price point
  const calculateGainFromRef = (price: number): string | null => {
    if (referenceClose === null || referenceClose === 0) return null;
    const gain = ((price - referenceClose) / referenceClose) * 100;
    const sign = gain >= 0 ? '+' : '';
    return `${sign}${gain.toFixed(2)}%`;
  };

  // Custom tooltip with percentage gain
  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null;
    
    const dataPoint = payload[0]?.payload;
    const price = payload[0]?.value;
    const gainStr = calculateGainFromRef(price);
    const dateStr = dataPoint?.date ?? '';
    
    return (
      <div className="bg-[rgba(10,10,15,0.95)] border border-white/10 rounded-lg px-3 py-2 shadow-xl">
        <p className="text-white/60 text-xs">{formatDate(dateStr, true)}</p>
        <p className="text-white font-medium text-sm mt-1">${price?.toFixed(2)}</p>
        {gainStr && (
          <p className={`text-xs mt-0.5 ${price >= (referenceClose ?? 0) ? 'text-green-400' : 'text-red-400'}`}>
            {gainStr} vs prev close
          </p>
        )}
      </div>
    );
  };

  if (!ticker) return null;

  const isPositive = (stock?.ytd_return ?? 0) >= 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal */}
      <div className="glass-card p-6 w-full max-w-4xl max-h-[90vh] overflow-y-auto relative z-10 fade-in">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-accent-cyan/30 to-accent-purple/30 flex items-center justify-center border border-white/10">
              <span className="font-bold text-white">{ticker.slice(0, 4)}</span>
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">{ticker}</h2>
              {stock && (
                <p className="text-white/50">
                  ${stock.current_price.toFixed(2)}
                  <span className={`ml-2 ${stock.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)} ({stock.change_pct.toFixed(2)}%)
                  </span>
                </p>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/10 transition-colors"
          >
            <X className="w-6 h-6 text-white/70" />
          </button>
        </div>

        {loading ? (
          <div className="h-80 flex items-center justify-center">
            <div className="w-10 h-10 border-3 border-white/20 border-t-accent-cyan rounded-full animate-spin" />
          </div>
        ) : stock ? (
          <>
            {/* Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <div className="flex items-center gap-2 text-white/50 text-sm mb-1">
                  <Calendar className="w-4 h-4" />
                  YTD Return
                </div>
                <div className="flex items-center gap-2">
                  {isPositive ? (
                    <TrendingUp className="w-5 h-5 text-green-400" />
                  ) : (
                    <TrendingDown className="w-5 h-5 text-red-400" />
                  )}
                  <span className={`text-xl font-bold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                    {isPositive ? '+' : ''}{stock.ytd_return.toFixed(2)}%
                  </span>
                </div>
              </div>

              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <div className="flex items-center gap-2 text-white/50 text-sm mb-1">
                  <BarChart2 className="w-4 h-4" />
                  Period Gain
                </div>
                {!isDataComplete ? (
                  <p className="text-lg font-bold text-yellow-500/70" title={`Data from ${actualStart || '?'}, expected from ${expectedStart || '?'}`}>
                    Loading...
                  </p>
                ) : periodGain !== null ? (
                  <div className="flex items-center gap-2">
                    {isAboveReference ? (
                      <TrendingUp className="w-5 h-5 text-green-400" />
                    ) : (
                      <TrendingDown className="w-5 h-5 text-red-400" />
                    )}
                    <span className={`text-xl font-bold ${isAboveReference ? 'text-green-400' : 'text-red-400'}`}>
                      {periodGain >= 0 ? '+' : ''}{periodGain.toFixed(2)}%
                    </span>
                  </div>
                ) : (
                  <p className="text-xl font-bold text-white/50">—</p>
                )}
              </div>

              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <p className="text-white/50 text-sm mb-1">52W High</p>
                <p className="text-xl font-bold text-white">
                  ${stock.high_52w?.toFixed(2) ?? '—'}
                </p>
              </div>

              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <p className="text-white/50 text-sm mb-1">52W Low</p>
                <p className="text-xl font-bold text-white">
                  ${stock.low_52w?.toFixed(2) ?? '—'}
                </p>
              </div>
            </div>

            {/* Period Selector */}
            <div className="mb-4">
              <ChartPeriodSelector selected={chartPeriod} onSelect={onChartPeriodChange} />
            </div>

            {/* Reference Close Info */}
            {referenceClose !== null && (
              <div className="mb-2 text-sm text-white/50">
                <span className="inline-flex items-center gap-2">
                  <span className="w-4 h-0 border-t-2 border-dashed border-white/40"></span>
                  Previous Close: ${referenceClose.toFixed(2)}
                </span>
              </div>
            )}

            {/* Chart */}
            <div className="h-80 bg-white/5 rounded-xl p-4 border border-white/10 relative">
              {chartLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/30 rounded-xl z-10">
                  <div className="w-8 h-8 border-2 border-white/20 border-t-accent-cyan rounded-full animate-spin" />
                </div>
              )}
              {/* Limited data notice */}
              {!chartLoading && processedHistory.length > 0 && processedHistory.length <= 3 && (
                <div className="absolute top-4 right-4 z-10 bg-yellow-500/20 border border-yellow-500/30 rounded-lg px-3 py-1.5">
                  <span className="text-yellow-400 text-xs">
                    Limited data ({processedHistory.length} {processedHistory.length === 1 ? 'point' : 'points'})
                  </span>
                </div>
              )}
              {/* Missing data indicator - gray left section */}
              {!isDataComplete && processedHistory.length > 0 && !chartLoading && (
                <div 
                  className="absolute left-4 top-4 bottom-4 bg-gradient-to-r from-gray-500/20 to-transparent pointer-events-none z-[1] rounded-l-lg"
                  style={{ width: '15%' }}
                  title={`Data starts from ${actualStart || 'unknown'}, expected from ${expectedStart || 'unknown'}`}
                >
                  <div className="h-full border-l-2 border-dashed border-gray-500/40 flex items-center justify-center">
                    <span className="text-gray-400/60 text-xs -rotate-90 whitespace-nowrap">Missing Data</span>
                  </div>
                </div>
              )}
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={processedHistory}>
                  <defs>
                    <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={lineColor} stopOpacity={0.3}/>
                      <stop offset="95%" stopColor={lineColor} stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="extendedHoursGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#06b6d4" stopOpacity={0.2}/>
                      <stop offset="100%" stopColor="#10b981" stopOpacity={0.05}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  
                  {/* X-axis: numeric for intraday, category for daily */}
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
                      dataKey="date"
                      stroke="rgba(255,255,255,0.3)"
                      tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 11 }}
                      tickFormatter={formatXAxis}
                      interval="preserveStartEnd"
                      tickCount={6}
                    />
                  )}
                  <YAxis 
                    stroke="rgba(255,255,255,0.3)"
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                    domain={yAxisDomain}
                    tickFormatter={(value) => `$${value.toFixed(0)}`}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  
                  {/* Extended hours shading (pre-market & after-hours) */}
                  {extendedHoursRanges.map((range, idx) => (
                    <ReferenceArea
                      key={idx}
                      x1={range.start}
                      x2={range.end}
                      fill="url(#extendedHoursGradient)"
                      strokeOpacity={0}
                    />
                  ))}
                  
                  {/* Reference Line - Previous Close */}
                  {referenceClose !== null && (
                    <ReferenceLine 
                      y={referenceClose} 
                      stroke="rgba(255,255,255,0.4)" 
                      strokeDasharray="4 4"
                      strokeWidth={1.5}
                    />
                  )}
                  
                  {/* SMA Reference Line (only for longer periods) */}
                  {!isIntraday && stock.sma_200 && (
                    <ReferenceLine 
                      y={stock.sma_200} 
                      stroke="#a855f7" 
                      strokeDasharray="5 5"
                      label={{ 
                        value: `SMA(200): $${stock.sma_200.toFixed(2)}`, 
                        fill: '#a855f7',
                        fontSize: 11,
                        position: 'right'
                      }}
                    />
                  )}
                  
                  <Area
                    type="monotone"
                    dataKey="close"
                    stroke="transparent"
                    fill={`url(#${gradientId})`}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="close"
                    stroke={lineColor}
                    strokeWidth={2}
                    dot={(props: any) => {
                      const { cx, cy, index } = props;
                      const isLastPoint = index === processedHistory.length - 1;
                      const showAllDots = processedHistory.length <= 5;
                      
                      if (!isLastPoint && !showAllDots) return null;
                      
                      // Render dot (with ping effect for last point using CSS animation)
                      return (
                        <g key={isLastPoint ? `dot-${index}-${pingKey}` : `dot-${index}`}>
                          {/* Ping circle on last point - only show when data updates */}
                          {isLastPoint && showPing && (
                            <circle 
                              cx={cx} 
                              cy={cy} 
                              r={5} 
                              fill={lineColor}
                              className="chart-ping-dot-large"
                              style={{ transformOrigin: `${cx}px ${cy}px` }}
                            />
                          )}
                          {/* Static dot */}
                          <circle
                            cx={cx}
                            cy={cy}
                            r={isLastPoint ? 5 : 4}
                            fill={lineColor}
                            stroke="#0a0a0f"
                            strokeWidth={2}
                          />
                        </g>
                      );
                    }}
                    activeDot={{ r: 6, fill: lineColor, stroke: '#0a0a0f', strokeWidth: 2 }}
                    isAnimationActive={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            {/* Legend */}
            <div className="mt-3 flex flex-wrap gap-4 text-xs text-white/50">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-white/40" style={{ borderStyle: 'dashed' }}></span>
                Previous Close
              </span>
              {!isIntraday && stock.sma_200 && (
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-0.5 bg-purple-500"></span>
                  SMA(200)
                </span>
              )}
              <span className="flex items-center gap-1.5">
                <span className={`w-3 h-0.5 ${isAboveReference ? 'bg-green-500' : 'bg-red-500'}`}></span>
                {isAboveReference ? 'Above' : 'Below'} previous close
              </span>
              {isIntraday && extendedHoursRanges.length > 0 && (
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-sm bg-gradient-to-b from-cyan-500/40 to-emerald-500/20"></span>
                  Pre/After Market
                </span>
              )}
              {!isDataComplete && (
                <span className="flex items-center gap-1.5 text-yellow-500/70">
                  <span className="w-3 h-3 rounded-sm bg-gradient-to-r from-gray-500/30 to-transparent border-l border-dashed border-gray-500/50"></span>
                  Incomplete Data
                </span>
              )}
            </div>
          </>
        ) : (
          <div className="h-80 flex items-center justify-center text-white/50">
            Failed to load stock data
          </div>
        )}
      </div>
    </div>
  );
}
