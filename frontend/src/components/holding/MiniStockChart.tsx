import { useMemo, useEffect, useRef, useState } from 'react';
import { RefreshCw, Sun, Moon, Sunrise } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer, Tooltip, ReferenceLine, YAxis, XAxis, ReferenceArea } from 'recharts';
import type { HistoryPoint } from '../../types';
import { calculateSupportResistance } from '../../services/technicalIndicators';
import { useDataCacheContext } from '../../context/DataCacheContext';


// Extended hours trading window (4 AM to 8 PM Eastern)
const EXTENDED_START_MINUTES = 4 * 60;    // 4:00 AM = 240 minutes
const EXTENDED_END_MINUTES = 20 * 60;     // 8:00 PM = 1200 minutes

// Regular market hours
const MARKET_OPEN = 9 * 60 + 30;  // 9:30 AM = 570 minutes
const MARKET_CLOSE = 16 * 60;     // 4:00 PM = 960 minutes

// Session change type
interface SessionChange {
  percent: number;
  value: number;
}

interface SessionChanges {
  preMarket: SessionChange | null;
  regular: SessionChange | null;
  afterHours: SessionChange | null;
}

function parseTimeToMinutes(timeStr: string): number {
  const [hours, minutes] = timeStr.split(':').map(Number);
  if (isNaN(hours) || isNaN(minutes)) return -1;
  return hours * 60 + minutes;
}

function isMarketHours(dateStr: string): boolean {
  if (!dateStr.includes(' ')) return true; // Daily data
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

interface MiniStockChartProps {
  holdingId: number;
  ticker: string;
  history: HistoryPoint[];
  referenceClose: number | null;
  isDataComplete?: boolean;
  expectedStart?: string | null;
  actualStart?: string | null;
  isLoading?: boolean;
  lastPricesFetched?: Date | null;
}

// Convert 24-hour time to 12-hour format
function formatTime12Hour(time24: string): string {
  const [hours, minutes] = time24.split(':').map(Number);
  if (isNaN(hours) || isNaN(minutes)) return time24;
  const ampm = hours >= 12 ? 'PM' : 'AM';
  const hours12 = hours % 12 || 12;
  return `${hours12}:${minutes.toString().padStart(2, '0')} ${ampm}`;
}

function formatTooltipDate(dateStr: string): string {
  if (!dateStr) return '';
  
  // Parse date parts manually to avoid timezone issues
  // Format expected: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM"
  const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const currentYear = new Date().getFullYear();
  
  if (dateStr.includes(' ')) {
    const [datePart, timePart] = dateStr.split(' ');
    const [year, month, day] = datePart.split('-').map(Number);
    if (isNaN(year) || isNaN(month) || isNaN(day)) return dateStr;
    // Include year if different from current year
    const formattedDate = year !== currentYear 
      ? `${year} ${monthNames[month - 1]} ${day}`
      : `${monthNames[month - 1]} ${day}`;
    const time12 = formatTime12Hour(timePart);
    return `${formattedDate} — ${time12}`;
  }
  
  const [year, month, day] = dateStr.split('-').map(Number);
  if (isNaN(year) || isNaN(month) || isNaN(day)) return dateStr;
  // Include year if different from current year
  return year !== currentYear 
    ? `${year} ${monthNames[month - 1]} ${day}`
    : `${monthNames[month - 1]} ${day}`;
}

export function MiniStockChart({
  holdingId,
  ticker,
  history,
  referenceClose,
  isDataComplete = true,
  expectedStart,
  actualStart,
  isLoading = false,
  lastPricesFetched
}: MiniStockChartProps) {
  // Get extended history for S/R calculations (6-month daily data)
  // Only use extended history for accuracy - don't fall back to chart history
  const { extendedHistoryData } = useDataCacheContext();
  const tickerUpper = ticker.toUpperCase();
  const extendedHistory = extendedHistoryData.get(tickerUpper)?.history ?? [];
  // Ping animation state - triggers when prices API returns
  const [pingKey, setPingKey] = useState(0);
  const [showPing, setShowPing] = useState(false);
  const prevTimestampRef = useRef<number | null>(null);
  const isFirstRender = useRef(true);
  
  // Trigger ping when prices are fetched (API returns), regardless of data change
  useEffect(() => {
    const currentTimestamp = lastPricesFetched?.getTime() ?? null;
    
    // Skip the first render
    if (isFirstRender.current) {
      isFirstRender.current = false;
      prevTimestampRef.current = currentTimestamp;
      return;
    }
    
    // Trigger ping when prices API returns with data
    if (currentTimestamp !== null && currentTimestamp !== prevTimestampRef.current) {
      setPingKey(prev => prev + 1);
      setShowPing(true);
      const timer = setTimeout(() => setShowPing(false), 1000);
      prevTimestampRef.current = currentTimestamp;
      return () => clearTimeout(timer);
    }
    
    prevTimestampRef.current = currentTimestamp;
  }, [lastPricesFetched]);
  const isIntraday = useMemo(() => isIntradayData(history), [history]);
  
  // Calculate session-specific changes (pre-market, regular, after-hours)
  const sessionChanges = useMemo((): SessionChanges | null => {
    if (!isIntraday || !history.length) return null;
    
    // Get the most recent trading day's data
    const latestDay = getTradingDay(history[history.length - 1].date);
    const todayData = history.filter(h => getTradingDay(h.date) === latestDay);
    
    if (!todayData.length) return null;
    
    // Find first and last prices for each session
    let preMarketFirst: number | null = null;
    let preMarketLast: number | null = null;
    let regularFirst: number | null = null;
    let regularLast: number | null = null;
    let afterHoursFirst: number | null = null;
    let afterHoursLast: number | null = null;
    
    for (const point of todayData) {
      const minutes = dateToMinutes(point.date);
      
      if (minutes < MARKET_OPEN) {
        // Pre-market (4 AM - 9:30 AM)
        if (preMarketFirst === null) preMarketFirst = point.close;
        preMarketLast = point.close;
      } else if (minutes < MARKET_CLOSE) {
        // Regular hours (9:30 AM - 4 PM)
        if (regularFirst === null) regularFirst = point.close;
        regularLast = point.close;
      } else {
        // After hours (4 PM - 8 PM)
        if (afterHoursFirst === null) afterHoursFirst = point.close;
        afterHoursLast = point.close;
      }
    }
    
    // Helper to calculate change
    const calcChange = (start: number | null, end: number | null): SessionChange | null => {
      if (start === null || end === null || start === 0) return null;
      return {
        percent: ((end - start) / start) * 100,
        value: end - start
      };
    };
    
    // Pre-market: from reference close to end of pre-market
    // If no pre-market data, use null
    const preMarketBase = referenceClose ?? preMarketFirst;
    const preMarket = (preMarketBase && preMarketLast) 
      ? calcChange(preMarketBase, preMarketLast)
      : null;
    
    // Regular: from first regular price to last regular price
    const regular = calcChange(regularFirst, regularLast);
    
    // After-hours: from last regular price to last after-hours price
    const afterHoursBase = regularLast ?? afterHoursFirst;
    const afterHours = (afterHoursBase && afterHoursLast)
      ? calcChange(afterHoursBase, afterHoursLast)
      : null;
    
    return { preMarket, regular, afterHours };
  }, [history, isIntraday, referenceClose]);
  
  // Calculate period gain
  const periodGain = useMemo(() => {
    if (!history.length || referenceClose === null || referenceClose === 0) return null;
    const latestClose = history[history.length - 1]?.close ?? 0;
    return ((latestClose - referenceClose) / referenceClose) * 100;
  }, [history, referenceClose]);

  const isAboveReference = periodGain !== null ? periodGain >= 0 : true;
  const lineColor = isAboveReference ? '#22c55e' : '#ef4444';

  // Calculate support/resistance levels using extended history (6-month daily data)
  // Only calculate when we have sufficient historical data for accuracy
  // Requires 60+ days of data for meaningful weekly prediction
  const supportResistanceLevels = useMemo(() => {
    if (!history.length) return null;
    if (extendedHistory.length < 60) {
      console.log(`[S/R ${tickerUpper}] Not enough extended history: ${extendedHistory.length} points (need 60+)`);
      return null;
    }

    const currentPrice = history[history.length - 1].close;
    console.log(`[S/R ${tickerUpper}] Calculating with ${extendedHistory.length} points, price=${currentPrice.toFixed(2)}`);
    return calculateSupportResistance(extendedHistory, currentPrice, 2, 20, 5, tickerUpper);
  }, [history, extendedHistory, tickerUpper]);

  // Calculate support/resistance levels for tooltip (wider range, shows even if off-chart)
  const supportResistanceLevelsForTooltip = useMemo(() => {
    if (!history.length) return null;
    if (extendedHistory.length < 60) return null;

    const currentPrice = history[history.length - 1].close;
    return calculateSupportResistance(extendedHistory, currentPrice, 1, 100, 5);
  }, [history, extendedHistory]);

  // Calculate Y-axis domain (don't include S/R levels to avoid squishing)
  const yAxisDomain = useMemo((): [number, number] | [string, string] => {
    if (!history.length) return ['auto', 'auto'];

    const closes = history.map(h => h.close).filter(c => c > 0);
    if (!closes.length) return ['auto', 'auto'];

    let min = Math.min(...closes);
    let max = Math.max(...closes);

    if (referenceClose !== null && referenceClose > 0) {
      min = Math.min(min, referenceClose);
      max = Math.max(max, referenceClose);
    }

    const padding = (max - min) * 0.1;
    return [min - padding, max + padding] as [number, number];
  }, [history, referenceClose]);

  // Process history with time-based positioning for intraday data
  const { processedHistory, extendedHoursRanges, xDomain } = useMemo(() => {
    if (!history.length) {
      return { 
        processedHistory: [], 
        extendedHoursRanges: [], 
        xDomain: [0, 100] as [number, number] 
      };
    }
    
    if (!isIntraday) {
      // For daily data, use simple index-based positioning
      const processed = history.map((h, idx) => ({
        ...h,
        xValue: idx,
        isExtendedHours: false
      }));
      // Ensure domain has width even with single data point
      const maxIdx = Math.max(history.length - 1, 1);
      return { 
        processedHistory: processed, 
        extendedHoursRanges: [], 
        xDomain: [0, maxIdx] as [number, number]
      };
    }
    
    // For intraday data, use time-based positioning
    // Get unique trading days in the data
    const tradingDays = [...new Set(history.map(h => getTradingDay(h.date)))].sort();
    const numDays = tradingDays.length;
    const minutesPerDay = EXTENDED_END_MINUTES - EXTENDED_START_MINUTES; // 960 minutes (4 AM to 8 PM)
    
    const processed = history.map((h) => {
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
    // This ensures the chart scale stays consistent regardless of current time
    const domain: [number, number] = [0, numDays * minutesPerDay];
    
    // Find extended hours ranges (for shading)
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
    
    return { processedHistory: processed, extendedHoursRanges: ranges, xDomain: domain };
  }, [history, isIntraday]);

  const calculateGainFromRef = (price: number): string | null => {
    if (referenceClose === null || referenceClose === 0) return null;
    const gain = ((price - referenceClose) / referenceClose) * 100;
    const sign = gain >= 0 ? '+' : '';
    return `${sign}${gain.toFixed(2)}%`;
  };

  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null;

    const price = payload[0]?.value;
    const gainStr = calculateGainFromRef(price);
    const dateStr = payload[0]?.payload?.date ?? '';

    // Find nearest support and resistance (use wider range for tooltip)
    const nearestResistance = supportResistanceLevelsForTooltip?.resistance[0];
    const nearestSupport = supportResistanceLevelsForTooltip?.support[0];

    // Check if these levels are shown on chart (within 20% range)
    const resistanceOnChart = supportResistanceLevels?.resistance.some(
      r => r.price === nearestResistance?.price
    );
    const supportOnChart = supportResistanceLevels?.support.some(
      s => s.price === nearestSupport?.price
    );

    const distToResistance = nearestResistance
      ? ((nearestResistance.price - price) / price * 100).toFixed(1)
      : null;
    const distToSupport = nearestSupport
      ? ((price - nearestSupport.price) / price * 100).toFixed(1)
      : null;

    return (
      <div className="bg-[rgba(10,10,15,0.95)] border border-white/10 rounded-lg px-3 py-2 text-xs z-50 relative">
        <p className="text-white/50">{formatTooltipDate(dateStr)}</p>
        <p className="text-white font-medium mt-1">${price?.toFixed(2)}</p>
        {gainStr && (
          <p className={`text-xs mt-0.5 ${price >= (referenceClose ?? 0) ? 'text-green-400' : 'text-red-400'}`}>
            {gainStr} vs prev close
          </p>
        )}
        {(nearestResistance || nearestSupport) && (
          <div className="mt-1.5 pt-1.5 border-t border-white/10 space-y-1">
            {nearestResistance && (
              <div className="text-orange-400/80">
                <div className="flex items-center gap-1">
                  <span>R: ${nearestResistance.price.toFixed(2)}</span>
                  <span className="text-white/50">(+{distToResistance}%)</span>
                  {!resistanceOnChart && <span className="text-white/40">(off chart)</span>}
                </div>
                <div className="flex items-center gap-1.5 text-[10px] text-white/50 mt-0.5">
                  <span className={`px-1 rounded ${
                    nearestResistance.strength >= 0.7 ? 'bg-orange-500/30 text-orange-300' :
                    nearestResistance.strength >= 0.5 ? 'bg-yellow-500/20 text-yellow-300/80' :
                    'bg-white/10 text-white/50'
                  }`}>
                    {Math.round(nearestResistance.strength * 100)}%
                  </span>
                  <span>{nearestResistance.touches}x touch{nearestResistance.touches !== 1 ? 'es' : ''}</span>
                  {nearestResistance.avgVolumeRatio !== undefined && nearestResistance.avgVolumeRatio > 0 && (
                    <span className={nearestResistance.avgVolumeRatio >= 1.5 ? 'text-yellow-400/70' : ''}>
                      {nearestResistance.avgVolumeRatio >= 1.5 ? '📊' : '•'} {nearestResistance.avgVolumeRatio.toFixed(1)}x vol
                    </span>
                  )}
                  {nearestResistance.rejectionCount !== undefined && nearestResistance.rejectionCount > 0 && (
                    <span className="text-orange-300/70">
                      ⚡{nearestResistance.rejectionCount} rej
                    </span>
                  )}
                  {nearestResistance.hasRoleReversal && (
                    <span className="text-purple-400/80" title="Role reversal (broken support became resistance)">
                      🔄 flip
                    </span>
                  )}
                </div>
              </div>
            )}
            {nearestSupport && (
              <div className="text-cyan-400/80">
                <div className="flex items-center gap-1">
                  <span>S: ${nearestSupport.price.toFixed(2)}</span>
                  <span className="text-white/50">(-{distToSupport}%)</span>
                  {!supportOnChart && <span className="text-white/40">(off chart)</span>}
                </div>
                <div className="flex items-center gap-1.5 text-[10px] text-white/50 mt-0.5">
                  <span className={`px-1 rounded ${
                    nearestSupport.strength >= 0.7 ? 'bg-cyan-500/30 text-cyan-300' :
                    nearestSupport.strength >= 0.5 ? 'bg-yellow-500/20 text-yellow-300/80' :
                    'bg-white/10 text-white/50'
                  }`}>
                    {Math.round(nearestSupport.strength * 100)}%
                  </span>
                  <span>{nearestSupport.touches}x touch{nearestSupport.touches !== 1 ? 'es' : ''}</span>
                  {nearestSupport.avgVolumeRatio !== undefined && nearestSupport.avgVolumeRatio > 0 && (
                    <span className={nearestSupport.avgVolumeRatio >= 1.5 ? 'text-yellow-400/70' : ''}>
                      {nearestSupport.avgVolumeRatio >= 1.5 ? '📊' : '•'} {nearestSupport.avgVolumeRatio.toFixed(1)}x vol
                    </span>
                  )}
                  {nearestSupport.rejectionCount !== undefined && nearestSupport.rejectionCount > 0 && (
                    <span className="text-cyan-300/70">
                      ⚡{nearestSupport.rejectionCount} rej
                    </span>
                  )}
                  {nearestSupport.hasRoleReversal && (
                    <span className="text-purple-400/80" title="Role reversal (broken resistance became support)">
                      🔄 flip
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <>
      <div className="h-28 mt-2 relative">
        {isLoading && (
          <div className="absolute top-0 right-0 z-10">
            <RefreshCw className="w-3 h-3 text-accent-cyan/60 animate-spin" />
          </div>
        )}
        
        {/* Missing data indicator */}
        {!isDataComplete && processedHistory.length > 0 && (
          <div 
            className="absolute left-0 top-0 bottom-0 bg-gradient-to-r from-gray-500/20 to-transparent pointer-events-none z-[1]"
            style={{ width: '20%' }}
            title={`Data starts from ${actualStart || 'unknown'}, expected from ${expectedStart || 'unknown'}`}
          >
            <div className="h-full border-l-2 border-dashed border-gray-500/40" />
          </div>
        )}
        
        {processedHistory.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={processedHistory}>
              <defs>
                <linearGradient id={`extendedHoursGradient-${holdingId}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#06b6d4" stopOpacity={0.15}/>
                  <stop offset="100%" stopColor="#10b981" stopOpacity={0.05}/>
                </linearGradient>
              </defs>
              
              {/* X-axis with time-based domain for intraday data */}
              <XAxis 
                dataKey="xValue" 
                type="number"
                domain={xDomain}
                hide 
              />
              <YAxis domain={yAxisDomain} hide />
              <Tooltip content={<CustomTooltip />} wrapperStyle={{ zIndex: 50 }} />
              
              {/* Extended hours shading */}
              {extendedHoursRanges.map((range, idx) => (
                <ReferenceArea
                  key={idx}
                  x1={range.start}
                  x2={range.end}
                  fill={`url(#extendedHoursGradient-${holdingId})`}
                  strokeOpacity={0}
                />
              ))}
              
              {referenceClose !== null && (
                <ReferenceLine
                  y={referenceClose}
                  stroke="rgba(255,255,255,0.35)"
                  strokeDasharray="3 3"
                  strokeWidth={1}
                />
              )}

              {/* Support/Resistance levels - strength affects visual weight */}
              {supportResistanceLevels?.resistance.map((level, idx) => {
                // Strength affects: line width (1-2), opacity (0.4-0.9), dash density
                const width = 1 + level.strength * 1;
                const opacity = 0.4 + level.strength * 0.5;
                // Stronger = more dots (shorter gaps): "2 6" (weak) to "3 2" (strong)
                const dashGap = Math.round(6 - level.strength * 4);
                const dashLength = Math.round(2 + level.strength * 1);
                return (
                  <ReferenceLine
                    key={`r-${idx}`}
                    y={level.price}
                    stroke="#f97316"
                    strokeDasharray={`${dashLength} ${dashGap}`}
                    strokeWidth={width}
                    strokeOpacity={opacity}
                    label={{
                      value: `R $${level.price.toFixed(2)}`,
                      position: 'right',
                      fill: '#f97316',
                      fontSize: 9,
                      opacity: opacity
                    }}
                  />
                );
              })}
              {supportResistanceLevels?.support.map((level, idx) => {
                const width = 1 + level.strength * 1;
                const opacity = 0.4 + level.strength * 0.5;
                const dashGap = Math.round(6 - level.strength * 4);
                const dashLength = Math.round(2 + level.strength * 1);
                return (
                  <ReferenceLine
                    key={`s-${idx}`}
                    y={level.price}
                    stroke="#06b6d4"
                    strokeDasharray={`${dashLength} ${dashGap}`}
                    strokeWidth={width}
                    strokeOpacity={opacity}
                    label={{
                      value: `S $${level.price.toFixed(2)}`,
                      position: 'right',
                      fill: '#06b6d4',
                      fontSize: 9,
                      opacity: opacity
                    }}
                  />
                );
              })}

              <Line
                type="monotone"
                dataKey="close"
                stroke={lineColor}
                strokeWidth={2}
                dot={(props: any) => {
                  const { cx, cy, index } = props;
                  const isLastPoint = index === processedHistory.length - 1;
                  
                  if (!isLastPoint) return <g key={`empty-${index}`} />;
                  
                  // Render the last point with ping effect using CSS animation
                  return (
                    <g key={`last-dot-${holdingId}-${pingKey}`}>
                      {/* Ping circle - only show when data updates */}
                      {showPing && (
                        <circle 
                          cx={cx} 
                          cy={cy} 
                          r={4} 
                          fill={lineColor}
                          className="chart-ping-dot"
                          style={{ transformOrigin: `${cx}px ${cy}px` }}
                        />
                      )}
                      {/* Static dot */}
                      <circle
                        cx={cx}
                        cy={cy}
                        r={3}
                        fill={lineColor}
                        stroke="#0a0a0f"
                        strokeWidth={1.5}
                      />
                    </g>
                  );
                }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : !isLoading ? (
          <div className="w-full h-full flex items-center justify-center text-white/30 text-xs">
            No chart data
          </div>
        ) : null}
      </div>

      {/* Session Changes Display */}
      {sessionChanges && (sessionChanges.preMarket || sessionChanges.regular || sessionChanges.afterHours) && (
        <div className="mt-2 grid grid-cols-3 gap-1.5 relative z-0">
          <SessionBadge 
            label="Pre" 
            icon={<Sunrise className="w-2.5 h-2.5" />}
            change={sessionChanges.preMarket} 
          />
          <SessionBadge 
            label="Mkt" 
            icon={<Sun className="w-2.5 h-2.5" />}
            change={sessionChanges.regular} 
          />
          <SessionBadge 
            label="AH" 
            icon={<Moon className="w-2.5 h-2.5" />}
            change={sessionChanges.afterHours} 
          />
        </div>
      )}

      {/* Legend */}
      {processedHistory.length > 0 && (
        <div className="mt-1 flex items-center gap-3 text-[10px] text-white/40 relative z-0">
          {referenceClose !== null && (
            <span className="inline-flex items-center gap-1">
              <span className="w-3 h-0 border-t border-dashed border-white/35"></span>
              Prev: ${referenceClose.toFixed(2)}
            </span>
          )}
          {extendedHoursRanges.length > 0 && (
            <span className="inline-flex items-center gap-1">
              <span className="w-3 h-2 rounded-sm bg-gradient-to-b from-cyan-500/30 to-emerald-500/20"></span>
              Extended
            </span>
          )}
          {supportResistanceLevels?.resistance.length ? (
            <span className="inline-flex items-center gap-1">
              <span className="w-3 h-0 border-t-2 border-dotted border-orange-500/80"></span>
              Resist
            </span>
          ) : null}
          {supportResistanceLevels?.support.length ? (
            <span className="inline-flex items-center gap-1">
              <span className="w-3 h-0 border-t-2 border-dotted border-cyan-500/80"></span>
              Support
            </span>
          ) : null}
          {!isDataComplete && (
            <span className="inline-flex items-center gap-1 text-yellow-500/70">
              <span className="w-3 h-2 rounded-sm bg-gradient-to-r from-gray-500/30 to-transparent border-l border-dashed border-gray-500/50"></span>
              Incomplete
            </span>
          )}
        </div>
      )}
    </>
  );
}

// Session change badge component
function SessionBadge({ 
  label, 
  icon, 
  change 
}: { 
  label: string; 
  icon: React.ReactNode; 
  change: SessionChange | null;
}) {
  // Format value as dollar amount
  const formatValue = (val: number) => {
    const sign = val >= 0 ? '+' : '';
    return `${sign}$${val.toFixed(2)}`;
  };
  
  // Empty state when no data for this session
  if (!change) {
    return (
      <div className="flex items-center justify-center gap-1 px-1.5 py-1 rounded text-[10px] bg-white/5 text-white/30 border border-white/5">
        <span className="opacity-50">{icon}</span>
        <span>{label}</span>
        <span>—</span>
      </div>
    );
  }
  
  const isPositive = change.percent >= 0;
  const sign = isPositive ? '+' : '';
  
  return (
    <div 
      className={`
        flex items-center justify-center gap-1 px-1.5 py-1 rounded text-[10px] font-medium
        ${isPositive 
          ? 'bg-green-500/15 text-green-400 border border-green-500/20' 
          : 'bg-red-500/15 text-red-400 border border-red-500/20'
        }
      `}
      title={`${label}: ${sign}${change.percent.toFixed(2)}% (${formatValue(change.value)})`}
    >
      <span className="opacity-70">{icon}</span>
      <span className="opacity-60">{label}</span>
      <span>{sign}{change.percent.toFixed(1)}%</span>
      <span className="opacity-50 text-[9px]">{formatValue(change.value)}</span>
    </div>
  );
}
