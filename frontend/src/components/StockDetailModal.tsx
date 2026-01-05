import { useEffect, useState, useMemo } from 'react';
import { X, TrendingUp, TrendingDown, Calendar, BarChart2 } from 'lucide-react';
import { 
  Line, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  CartesianGrid, ReferenceLine, Area, ComposedChart, ReferenceArea 
} from 'recharts';
import type { StockData, HistoryPoint } from '../types';
import * as api from '../services/api';
import { ChartPeriodSelector, type ChartPeriod } from './ChartPeriodSelector';

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

interface StockDetailModalProps {
  ticker: string | null;
  onClose: () => void;
}

export function StockDetailModal({ ticker, onClose }: StockDetailModalProps) {
  const [stock, setStock] = useState<StockData | null>(null);
  const [loading, setLoading] = useState(false);
  const [period, setPeriod] = useState<ChartPeriod>('1d');
  const [chartHistory, setChartHistory] = useState<HistoryPoint[]>([]);
  const [referenceClose, setReferenceClose] = useState<number | null>(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [isDataComplete, setIsDataComplete] = useState(true);
  const [expectedStart, setExpectedStart] = useState<string | null>(null);
  const [actualStart, setActualStart] = useState<string | null>(null);

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

    const fetchHistory = async () => {
      setChartLoading(true);
      try {
        const result = await api.getStockHistory(ticker, period);
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
  }, [ticker, period]);

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

  // Check if this is an intraday period
  const isIntraday = ['1d', '3d', '1w'].includes(period);

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

  // Process data to find extended hours ranges (for intraday charts)
  const extendedHoursRanges = useMemo(() => {
    if (!isIntraday || !chartHistory.length) return [];
    
    const ranges: { startDate: string; endDate: string }[] = [];
    let rangeStart: string | null = null;
    
    for (let i = 0; i < chartHistory.length; i++) {
      const point = chartHistory[i];
      const isExtended = !isMarketHours(point.date);
      
      if (isExtended) {
        if (rangeStart === null) rangeStart = point.date;
      } else {
        if (rangeStart !== null) {
          ranges.push({ startDate: rangeStart, endDate: chartHistory[i - 1].date });
          rangeStart = null;
        }
      }
    }
    // Close any open range
    if (rangeStart !== null) {
      ranges.push({ startDate: rangeStart, endDate: chartHistory[chartHistory.length - 1].date });
    }
    
    return ranges;
  }, [chartHistory, isIntraday]);

  // Convert 24-hour time (HH:MM) to 12-hour format
  const formatTime12Hour = (time24: string): string => {
    const [hours, minutes] = time24.split(':').map(Number);
    if (isNaN(hours) || isNaN(minutes)) return time24;
    const ampm = hours >= 12 ? 'PM' : 'AM';
    const hours12 = hours % 12 || 12;
    return `${hours12}:${minutes.toString().padStart(2, '0')} ${ampm}`;
  };

  // Format date/time properly
  const formatDate = (dateStr: string, includeTime: boolean): string => {
    if (!dateStr) return '';
    
    // Check if it contains time (format: "YYYY-MM-DD HH:MM")
    if (dateStr.includes(' ')) {
      const [datePart, timePart] = dateStr.split(' ');
      try {
        const date = new Date(datePart);
        if (isNaN(date.getTime())) return dateStr;
        const time12 = formatTime12Hour(timePart);
        if (includeTime) {
          const formattedDate = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
          return `${formattedDate} ${time12}`;
        }
        return time12; // Just time for X-axis
      } catch {
        return dateStr;
      }
    }
    
    // Daily format (YYYY-MM-DD)
    try {
      const date = new Date(dateStr);
      if (isNaN(date.getTime())) return dateStr;
      if (includeTime) {
        return date.toLocaleDateString('en-US', { 
          weekday: 'short', 
          month: 'short', 
          day: 'numeric',
          year: 'numeric'
        });
      }
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    } catch {
      return dateStr;
    }
  };

  // Format X-axis labels based on period
  const formatXAxis = (dateStr: string) => {
    return formatDate(dateStr, false);
  };

  // Calculate gain % for a specific price point
  const calculateGainFromRef = (price: number): string | null => {
    if (referenceClose === null || referenceClose === 0) return null;
    const gain = ((price - referenceClose) / referenceClose) * 100;
    const sign = gain >= 0 ? '+' : '';
    return `${sign}${gain.toFixed(2)}%`;
  };

  // Custom tooltip with percentage gain
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;
    
    const price = payload[0]?.value;
    const gainStr = calculateGainFromRef(price);
    
    return (
      <div className="bg-[rgba(10,10,15,0.95)] border border-white/10 rounded-lg px-3 py-2 shadow-xl">
        <p className="text-white/60 text-xs mb-1">{formatDate(String(label), true)}</p>
        <p className="text-white font-medium text-sm">${price?.toFixed(2)}</p>
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
  const isAboveSMA = (stock?.price_vs_sma ?? 0) >= 0;

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
              <ChartPeriodSelector selected={period} onSelect={setPeriod} />
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
              {/* Missing data indicator - gray left section */}
              {!isDataComplete && chartHistory.length > 0 && !chartLoading && (
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
                <ComposedChart data={chartHistory}>
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
                  <XAxis 
                    dataKey="date" 
                    stroke="rgba(255,255,255,0.3)"
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 11 }}
                    tickFormatter={formatXAxis}
                    interval="preserveStartEnd"
                    minTickGap={isIntraday ? 60 : 40}
                  />
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
                      x1={range.startDate}
                      x2={range.endDate}
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
                    dot={false}
                    activeDot={{ r: 5, fill: lineColor, stroke: '#0a0a0f', strokeWidth: 2 }}
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
