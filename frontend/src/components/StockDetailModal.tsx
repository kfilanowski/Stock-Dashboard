import { useEffect, useState, useMemo, useRef, useCallback } from 'react';
import { X, TrendingUp, TrendingDown, Calendar, BarChart2, Sun, Moon, Sunrise, Clock, ArrowUpRight, ArrowDownRight, ChevronDown, Check } from 'lucide-react';
import { 
  Line, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  CartesianGrid, ReferenceLine, Area, ComposedChart, ReferenceArea 
} from 'recharts';
import type { StockData, HistoryPoint } from '../types';
import * as api from '../services/api';
import { ChartPeriodSelector, type ChartPeriod } from './ChartPeriodSelector';
import { useHistoryForPeriod } from '../hooks/useDataCache';
import { calculateSMA, calculateEMA, calculateBollingerBands, calculateVWAP } from '../services/technicalIndicators';

// ============================================================================
// Technical Indicator Configuration
// ============================================================================

type IndicatorKey = 'sma20' | 'sma50' | 'sma200' | 'ema12' | 'ema26' | 'bbUpper' | 'bbMiddle' | 'bbLower' | 'vwap';

interface IndicatorConfig {
  key: IndicatorKey;
  label: string;
  color: string;
  strokeDasharray?: string;
  group: 'moving_averages' | 'bollinger' | 'volume';
  minPeriod: number; // Minimum data points needed
}

const INDICATOR_CONFIGS: IndicatorConfig[] = [
  // Moving Averages
  { key: 'sma20', label: 'SMA 20', color: '#f59e0b', group: 'moving_averages', minPeriod: 20 },
  { key: 'sma50', label: 'SMA 50', color: '#3b82f6', group: 'moving_averages', minPeriod: 50 },
  { key: 'sma200', label: 'SMA 200', color: '#a855f7', group: 'moving_averages', minPeriod: 200 },
  { key: 'ema12', label: 'EMA 12', color: '#10b981', strokeDasharray: '4 2', group: 'moving_averages', minPeriod: 12 },
  { key: 'ema26', label: 'EMA 26', color: '#06b6d4', strokeDasharray: '4 2', group: 'moving_averages', minPeriod: 26 },
  // Bollinger Bands
  { key: 'bbUpper', label: 'BB Upper', color: '#ec4899', strokeDasharray: '2 2', group: 'bollinger', minPeriod: 20 },
  { key: 'bbMiddle', label: 'BB Middle', color: '#ec4899', group: 'bollinger', minPeriod: 20 },
  { key: 'bbLower', label: 'BB Lower', color: '#ec4899', strokeDasharray: '2 2', group: 'bollinger', minPeriod: 20 },
  // Volume-based
  { key: 'vwap', label: 'VWAP', color: '#eab308', strokeDasharray: '6 3', group: 'volume', minPeriod: 5 },
];

// Extended hours trading window (4 AM to 8 PM Eastern)
const EXTENDED_START_MINUTES = 4 * 60;    // 4:00 AM = 240 minutes
const EXTENDED_END_MINUTES = 20 * 60;     // 8:00 PM = 1200 minutes

// Market hours in Eastern Time (24-hour format)
const MARKET_OPEN = 9 * 60 + 30;  // 9:30 AM = 570 minutes
const MARKET_CLOSE = 16 * 60;     // 4:00 PM = 960 minutes

// Session change types
interface SessionChange {
  percent: number;
  value: number;
}

interface SessionChanges {
  preMarket: SessionChange | null;
  regular: SessionChange | null;
  afterHours: SessionChange | null;
}

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
// For multi-day views (3d/1w), tradingDays array is used to get actual day names
function minutesToTimeLabel(
  minutes: number, 
  tradingDays?: string[],
  showDayNameOnly?: boolean
): string {
  const minutesPerDay = EXTENDED_END_MINUTES - EXTENDED_START_MINUTES; // 960 (4AM to 8PM)
  const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  
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
  
  // For 3-day and 1-week views, show day name only
  if (showDayNameOnly && tradingDays && tradingDays[dayIndex]) {
    const dateStr = tradingDays[dayIndex];
    const [year, month, day] = dateStr.split('-').map(Number);
    const date = new Date(year, month - 1, day);
    return dayNames[date.getDay()];
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
  
  return timeStr;
}

interface StockDetailModalProps {
  ticker: string | null;
  onClose: () => void;
  /** Initial chart period to display when the modal opens */
  initialChartPeriod?: ChartPeriod;
  lastPricesFetched?: Date | null;
  // Live data from parent (updated when prices refresh)
  holding?: {
    current_price?: number;
    ytd_return?: number;
    sma_200?: number;
    price_vs_sma?: number;
  } | null;
}

export function StockDetailModal({ 
  ticker, 
  onClose, 
  initialChartPeriod = '1d',
  lastPricesFetched,
  holding
}: StockDetailModalProps) {
  const [stock, setStock] = useState<StockData | null>(null);
  const [loading, setLoading] = useState(false);
  // Local chart period state - isolated from parent dashboard
  const [chartPeriod, setChartPeriod] = useState<ChartPeriod>(initialChartPeriod);
  // Technical indicators toggle state
  const [enabledIndicators, setEnabledIndicators] = useState<Set<IndicatorKey>>(new Set());
  const [indicatorMenuOpen, setIndicatorMenuOpen] = useState(false);
  const indicatorMenuRef = useRef<HTMLDivElement>(null);
  
  // Reset chart period to initial value when modal opens for a new ticker
  useEffect(() => {
    if (ticker) {
      setChartPeriod(initialChartPeriod);
    }
  }, [ticker, initialChartPeriod]);
  
  // Use centralized cache for chart history
  // This hook will use cached data if available or fetch if needed
  const { data: cachedChartData, isLoading: chartLoading } = useHistoryForPeriod(ticker, chartPeriod);
  
  // Extract chart data from cache
  const chartHistory = cachedChartData?.history ?? [];
  const referenceClose = cachedChartData?.referenceClose ?? null;
  const isDataComplete = cachedChartData?.isComplete ?? true;
  const expectedStart = cachedChartData?.expectedStart ?? null;
  const actualStart = cachedChartData?.actualStart ?? null;
  
  // Ping animation state - triggers when prices API returns
  const [pingKey, setPingKey] = useState(0);
  const [showPing, setShowPing] = useState(false);
  const prevTimestampRef = useRef<number | null>(null);
  const isFirstPricesFetch = useRef(true);
  
  // Chart range selection state
  const [selectionStart, setSelectionStart] = useState<number | null>(null);
  const [selectionEnd, setSelectionEnd] = useState<number | null>(null);
  const [isSelecting, setIsSelecting] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState<{ x: number; y: number } | null>(null);
  const chartContainerRef = useRef<HTMLDivElement>(null);

  // Fetch stock data on mount only (not on every price update - live data comes from holding prop)
  useEffect(() => {
    if (!ticker) {
      setStock(null);
      return;
    }

    const fetchStock = async () => {
      // Only show loading spinner on initial fetch
      if (!stock) setLoading(true);
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
  }, [ticker]); // Only fetch once when ticker changes - live updates come via holding prop

  // Clear selection when chart period changes
  useEffect(() => {
    setSelectionStart(null);
    setSelectionEnd(null);
    setIsSelecting(false);
    setTooltipPosition(null);
  }, [chartPeriod]);

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

  // Close indicator menu when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (indicatorMenuRef.current && !indicatorMenuRef.current.contains(event.target as Node)) {
        setIndicatorMenuOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Toggle indicator on/off
  const toggleIndicator = useCallback((key: IndicatorKey) => {
    setEnabledIndicators(prev => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }, []);

  // Toggle entire group of indicators
  const toggleIndicatorGroup = useCallback((group: 'moving_averages' | 'bollinger' | 'volume') => {
    const groupIndicators = INDICATOR_CONFIGS.filter(c => c.group === group).map(c => c.key);
    setEnabledIndicators(prev => {
      const next = new Set(prev);
      const allEnabled = groupIndicators.every(k => next.has(k));
      if (allEnabled) {
        // Disable all in group
        groupIndicators.forEach(k => next.delete(k));
      } else {
        // Enable all in group
        groupIndicators.forEach(k => next.add(k));
      }
      return next;
    });
  }, []);
  
  // Merge live holding data with fetched stock data
  const displayStock = useMemo(() => {
    if (!stock) return null;
    return {
      ...stock,
      // Override with live data from holding if available
      current_price: holding?.current_price ?? stock.current_price,
      ytd_return: holding?.ytd_return ?? stock.ytd_return,
      sma_200: holding?.sma_200 ?? stock.sma_200,
      price_vs_sma: holding?.price_vs_sma ?? stock.price_vs_sma,
      // Recalculate change based on live price
      change: (holding?.current_price ?? stock.current_price) - stock.previous_close,
      change_pct: stock.previous_close > 0 
        ? (((holding?.current_price ?? stock.current_price) - stock.previous_close) / stock.previous_close) * 100
        : 0
    };
  }, [stock, holding]);

  // Check if this is an intraday period
  const isIntraday = ['1d', '3d', '1w'].includes(chartPeriod);
  const hasIntradayData = useMemo(() => isIntradayData(chartHistory), [chartHistory]);

  // Calculate rolling technical indicators for each chart point
  const chartIndicators = useMemo((): Record<IndicatorKey, (number | null)[]> => {
    const emptyResult: Record<IndicatorKey, (number | null)[]> = {
      sma20: [],
      sma50: [],
      sma200: [],
      ema12: [],
      ema26: [],
      bbUpper: [],
      bbMiddle: [],
      bbLower: [],
      vwap: [],
    };
    
    if (!chartHistory.length) return emptyResult;
    
    const prices = chartHistory.map(h => h.close).filter(p => p > 0);
    const result = { ...emptyResult };
    
    // Calculate rolling values for each point
    for (let i = 0; i < chartHistory.length; i++) {
      const slice = prices.slice(0, i + 1);
      const historySlice = chartHistory.slice(0, i + 1);
      
      // SMAs
      result.sma20.push(slice.length >= 20 ? calculateSMA(slice, 20) : null);
      result.sma50.push(slice.length >= 50 ? calculateSMA(slice, 50) : null);
      result.sma200.push(slice.length >= 200 ? calculateSMA(slice, 200) : null);
      
      // EMAs
      result.ema12.push(slice.length >= 12 ? calculateEMA(slice, 12) : null);
      result.ema26.push(slice.length >= 26 ? calculateEMA(slice, 26) : null);
      
      // Bollinger Bands
      const bb = slice.length >= 20 ? calculateBollingerBands(historySlice, 20) : null;
      result.bbUpper.push(bb?.upper ?? null);
      result.bbMiddle.push(bb?.middle ?? null);
      result.bbLower.push(bb?.lower ?? null);
      
      // VWAP (uses volume, so use full history slice)
      const vwap = historySlice.length >= 5 ? calculateVWAP(historySlice, 20) : null;
      result.vwap.push(vwap?.value ?? null);
    }
    
    return result;
  }, [chartHistory]);

  // Get available indicators based on data length
  const availableIndicators = useMemo(() => {
    return INDICATOR_CONFIGS.filter(config => chartHistory.length >= config.minPeriod);
  }, [chartHistory.length]);

  // Calculate session-specific changes (pre-market, regular, after-hours)
  const sessionChanges = useMemo((): SessionChanges | null => {
    if (!hasIntradayData || !chartHistory.length) return null;
    
    // Get the most recent trading day's data
    const latestDay = getTradingDay(chartHistory[chartHistory.length - 1].date);
    const todayData = chartHistory.filter(h => getTradingDay(h.date) === latestDay);
    
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
  }, [chartHistory, hasIntradayData, referenceClose]);

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

  // Calculate Y-axis domain to include reference close and enabled indicators
  const yAxisDomain = useMemo((): [number, number] | [string, string] => {
    if (!chartHistory.length) return ['auto', 'auto'];
    
    const closes = chartHistory.map(h => h.close).filter(c => c > 0);
    if (!closes.length) return ['auto', 'auto'];
    
    let min = Math.min(...closes);
    let max = Math.max(...closes);
    
    // Include reference close in the domain
    if (referenceClose !== null && referenceClose > 0) {
      min = Math.min(min, referenceClose);
      max = Math.max(max, referenceClose);
    }
    
    // Include enabled indicators in the domain
    enabledIndicators.forEach(key => {
      const values = chartIndicators[key];
      if (values) {
        const validValues = values.filter((v): v is number => v !== null && v > 0);
        if (validValues.length) {
          min = Math.min(min, ...validValues);
          max = Math.max(max, ...validValues);
        }
      }
    });
    
    // Add 5% padding
    const padding = (max - min) * 0.05;
    return [min - padding, max + padding] as [number, number];
  }, [chartHistory, referenceClose, enabledIndicators, chartIndicators]);

  // Process chart data with time-based positioning for intraday
  const { processedHistory, extendedHoursRanges, xDomain, numTradingDays, tradingDays } = useMemo(() => {
    if (!chartHistory.length) {
      return { 
        processedHistory: [], 
        extendedHoursRanges: [], 
        xDomain: [0, 100] as [number, number],
        numTradingDays: 1,
        tradingDays: [] as string[]
      };
    }
    
    // Helper to add indicator values to a data point
    const addIndicatorValues = (h: HistoryPoint, idx: number) => ({
      ...h,
      sma20: chartIndicators.sma20?.[idx] ?? null,
      sma50: chartIndicators.sma50?.[idx] ?? null,
      sma200: chartIndicators.sma200?.[idx] ?? null,
      ema12: chartIndicators.ema12?.[idx] ?? null,
      ema26: chartIndicators.ema26?.[idx] ?? null,
      bbUpper: chartIndicators.bbUpper?.[idx] ?? null,
      bbMiddle: chartIndicators.bbMiddle?.[idx] ?? null,
      bbLower: chartIndicators.bbLower?.[idx] ?? null,
      vwap: chartIndicators.vwap?.[idx] ?? null,
    });
    
    if (!hasIntradayData) {
      // For daily data, use simple index-based positioning
      const processed = chartHistory.map((h, idx) => ({
        ...addIndicatorValues(h, idx),
        xValue: idx,
        isExtendedHours: false
      }));
      // Ensure domain has width even with single data point
      const maxIdx = Math.max(chartHistory.length - 1, 1);
      return { 
        processedHistory: processed, 
        extendedHoursRanges: [], 
        xDomain: [0, maxIdx] as [number, number],
        numTradingDays: 0,
        tradingDays: [] as string[]
      };
    }
    
    // For intraday data, use time-based positioning
    const days = [...new Set(chartHistory.map(h => getTradingDay(h.date)))].sort();
    const numDays = days.length;
    const minutesPerDay = EXTENDED_END_MINUTES - EXTENDED_START_MINUTES; // 960 minutes (4 AM to 8 PM)
    
    const processed = chartHistory.map((h, idx) => {
      const dayIndex = days.indexOf(getTradingDay(h.date));
      const minuteOfDay = dateToMinutes(h.date);
      // Position within the full time range: dayIndex * dayWidth + position within day
      const xValue = (dayIndex * minutesPerDay) + (minuteOfDay - EXTENDED_START_MINUTES);
      
      return {
        ...addIndicatorValues(h, idx),
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
    
    return { processedHistory: processed, extendedHoursRanges: ranges, xDomain: domain, numTradingDays: numDays, tradingDays: days };
  }, [chartHistory, hasIntradayData, chartIndicators]);
  
  // Chart selection mouse handlers
  const handleChartMouseDown = useCallback((e: any) => {
    if (!e || !e.activePayload || !e.activePayload[0]) return;
    const dataPoint = e.activePayload[0].payload;
    const xValue = hasIntradayData ? dataPoint.xValue : processedHistory.findIndex((p: any) => p.date === dataPoint.date);
    
    setSelectionStart(xValue);
    setSelectionEnd(xValue);
    setIsSelecting(true);
    setTooltipPosition(null);
  }, [hasIntradayData, processedHistory]);
  
  const handleChartMouseMove = useCallback((e: any) => {
    if (!isSelecting || !e || !e.activePayload || !e.activePayload[0]) return;
    const dataPoint = e.activePayload[0].payload;
    const xValue = hasIntradayData ? dataPoint.xValue : processedHistory.findIndex((p: any) => p.date === dataPoint.date);
    
    setSelectionEnd(xValue);
  }, [isSelecting, hasIntradayData, processedHistory]);
  
  const handleChartMouseUp = useCallback((e: any) => {
    if (!isSelecting) return;
    setIsSelecting(false);
    
    // Only show tooltip if there's a meaningful selection (start != end)
    if (selectionStart !== null && selectionEnd !== null && selectionStart !== selectionEnd) {
      // Get mouse position relative to chart container for tooltip positioning
      if (chartContainerRef.current && e?.chartX !== undefined && e?.chartY !== undefined) {
        const rect = chartContainerRef.current.getBoundingClientRect();
        setTooltipPosition({
          x: Math.min(e.chartX + 20, rect.width - 220), // Keep tooltip within bounds
          y: Math.max(e.chartY - 100, 10)
        });
      }
    } else {
      // Single click - clear selection
      setSelectionStart(null);
      setSelectionEnd(null);
      setTooltipPosition(null);
    }
  }, [isSelecting, selectionStart, selectionEnd]);
  
  const handleChartMouseLeave = useCallback(() => {
    if (isSelecting) {
      setIsSelecting(false);
      // Keep the selection if meaningful, otherwise clear
      if (selectionStart === selectionEnd) {
        setSelectionStart(null);
        setSelectionEnd(null);
      }
    }
  }, [isSelecting, selectionStart, selectionEnd]);
  
  const clearSelection = useCallback(() => {
    setSelectionStart(null);
    setSelectionEnd(null);
    setTooltipPosition(null);
  }, []);
  
  // Calculate selection statistics
  interface SelectionStats {
    startDate: string;
    endDate: string;
    startPrice: number;
    endPrice: number;
    priceChange: number;
    percentChange: number;
    rangeHigh: number;
    rangeLow: number;
    totalVolume: number;
    numPoints: number;
  }
  
  const selectionStats = useMemo((): SelectionStats | null => {
    if (selectionStart === null || selectionEnd === null || selectionStart === selectionEnd) {
      return null;
    }
    
    const minX = Math.min(selectionStart, selectionEnd);
    const maxX = Math.max(selectionStart, selectionEnd);
    
    // Filter points within the selection range
    const selectedPoints = processedHistory.filter((p: any) => {
      const xVal = hasIntradayData ? p.xValue : processedHistory.indexOf(p);
      return xVal >= minX && xVal <= maxX;
    });
    
    if (selectedPoints.length < 2) return null;
    
    const startPoint = selectedPoints[0];
    const endPoint = selectedPoints[selectedPoints.length - 1];
    
    const startPrice = startPoint.close;
    const endPrice = endPoint.close;
    const priceChange = endPrice - startPrice;
    const percentChange = startPrice > 0 ? ((endPrice - startPrice) / startPrice) * 100 : 0;
    
    const rangeHigh = Math.max(...selectedPoints.map((p: any) => p.high));
    const rangeLow = Math.min(...selectedPoints.map((p: any) => p.low));
    const totalVolume = selectedPoints.reduce((sum: number, p: any) => sum + (p.volume || 0), 0);
    
    return {
      startDate: startPoint.date,
      endDate: endPoint.date,
      startPrice,
      endPrice,
      priceChange,
      percentChange,
      rangeHigh,
      rangeLow,
      totalVolume,
      numPoints: selectedPoints.length
    };
  }, [selectionStart, selectionEnd, processedHistory, hasIntradayData]);

  // Generate nice tick values for intraday X-axis
  const xAxisTicks = useMemo(() => {
    if (!hasIntradayData || numTradingDays === 0) return undefined;
    
    const minutesPerDay = EXTENDED_END_MINUTES - EXTENDED_START_MINUTES; // 960 minutes
    const ticks: number[] = [];
    
    if (numTradingDays === 1) {
      // 1 day: 4AM, 6AM, 8AM, 10AM, 12PM, 2PM, 4PM, 6PM, 8PM
      [4, 6, 8, 10, 12, 14, 16, 18, 20].forEach(hour => {
        ticks.push((hour * 60) - EXTENDED_START_MINUTES);
      });
    } else {
      // 3 days or 1 week: One tick per day at 12PM (noon) for day name display
      for (let day = 0; day < numTradingDays; day++) {
        ticks.push(day * minutesPerDay + (12 * 60 - EXTENDED_START_MINUTES)); // 12PM
      }
    }
    
    return ticks;
  }, [hasIntradayData, numTradingDays]);
  
  // Generate tick values for daily (non-intraday) data based on chart period
  const dailyAxisTicks = useMemo(() => {
    if (hasIntradayData || !chartHistory.length) return undefined;
    
    const dates = chartHistory.map(h => h.date.split(' ')[0]); // Extract just date part
    const uniqueDates = [...new Set(dates)];
    
    // Helper to get first occurrence of each unique month (year-month combo)
    const getMonthStarts = (): string[] => {
      const monthStarts: string[] = [];
      let lastYearMonth = '';
      for (const date of uniqueDates) {
        const [year, month] = date.split('-');
        const yearMonth = `${year}-${month}`;
        if (yearMonth !== lastYearMonth) {
          monthStarts.push(date);
          lastYearMonth = yearMonth;
        }
      }
      return monthStarts;
    };
    
    // Helper to get first of each year from dates
    const getYearStarts = (): string[] => {
      const yearStarts: string[] = [];
      let lastYear = -1;
      for (const date of uniqueDates) {
        const year = parseInt(date.split('-')[0]);
        if (year !== lastYear) {
          yearStarts.push(date);
          lastYear = year;
        }
      }
      return yearStarts;
    };
    
    // Helper to get weekly ticks (every ~7 days)
    const getWeeklyTicks = (): string[] => {
      const weekly: string[] = [];
      for (let i = 0; i < uniqueDates.length; i += 5) { // ~5 trading days per week
        weekly.push(uniqueDates[i]);
      }
      return weekly;
    };
    
    switch (chartPeriod) {
      case '3mo':
        // One tick per week (day numbers)
        return getWeeklyTicks();
      case '6mo':
        // One label per month (should be ~6 months)
        return getMonthStarts();
      case 'ytd':
        // One label per month
        return getMonthStarts();
      case '1y':
        // One label per month (should be ~12 months)
        return getMonthStarts();
      case '2y':
        // One label per month
        return getMonthStarts();
      case '5y':
        // One label per year
        return getYearStarts();
      default:
        return undefined; // Let Recharts decide
    }
  }, [hasIntradayData, chartHistory, chartPeriod]);

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

  // Format X-axis labels based on chart period
  const formatXAxis = (value: number | string) => {
    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    
    // Helper to extract date parts from various formats
    const extractDateParts = (val: number | string): { year: number; month: number; day: number } | null => {
      if (typeof val === 'string') {
        // Format: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM"
        const datePart = val.includes(' ') ? val.split(' ')[0] : val;
        const [year, month, day] = datePart.split('-').map(Number);
        if (!isNaN(year) && !isNaN(month) && !isNaN(day)) {
          return { year, month, day };
        }
      } else if (typeof val === 'number' && tradingDays.length > 0) {
        // For intraday data, extract from tradingDays using xValue
        const minutesPerDay = EXTENDED_END_MINUTES - EXTENDED_START_MINUTES;
        const dayIndex = Math.floor(val / minutesPerDay);
        if (tradingDays[dayIndex]) {
          const [year, month, day] = tradingDays[dayIndex].split('-').map(Number);
          return { year, month, day };
        }
      }
      return null;
    };
    
    // 1 day: Show time labels
    if (chartPeriod === '1d' && hasIntradayData && typeof value === 'number') {
      return minutesToTimeLabel(value, tradingDays, false);
    }
    
    // 3 days or 1 week: Show day names (Mon, Tue, Wed)
    if (['3d', '1w'].includes(chartPeriod) && hasIntradayData && typeof value === 'number') {
      return minutesToTimeLabel(value, tradingDays, true);
    }
    
    // 1 month or 3 months: Show day number only (7, 14, 21, 28)
    if (['1mo', '3mo'].includes(chartPeriod)) {
      const parts = extractDateParts(value);
      if (parts) return `${parts.day}`;
    }
    
    // 6 months, YTD, 1 year: Show month name only (Jan, Feb, Mar)
    if (['6mo', 'ytd', '1y'].includes(chartPeriod)) {
      const parts = extractDateParts(value);
      if (parts) return monthNames[parts.month - 1];
    }
    
    // 2 years: Show month number (1, 2, 3, etc.)
    if (chartPeriod === '2y') {
      const parts = extractDateParts(value);
      if (parts) return `${parts.month}`;
    }
    
    // 5 years: Show 4-digit year
    if (chartPeriod === '5y') {
      const parts = extractDateParts(value);
      if (parts) return `${parts.year}`;
    }
    
    // Default fallback
    const parts = extractDateParts(value);
    if (parts) return `${monthNames[parts.month - 1]} ${parts.day}`;
    
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

  const isPositive = (displayStock?.ytd_return ?? 0) >= 0;

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
              {displayStock && (
                <p className="text-white/50">
                  ${displayStock.current_price.toFixed(2)}
                  <span className={`ml-2 ${displayStock.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {displayStock.change >= 0 ? '+' : ''}{displayStock.change.toFixed(2)} ({displayStock.change_pct.toFixed(2)}%)
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
        ) : displayStock ? (
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
                    {isPositive ? '+' : ''}{displayStock.ytd_return.toFixed(2)}%
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
                  ${displayStock.high_52w?.toFixed(2) ?? '—'}
                </p>
              </div>

              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <p className="text-white/50 text-sm mb-1">52W Low</p>
                <p className="text-xl font-bold text-white">
                  ${displayStock.low_52w?.toFixed(2) ?? '—'}
                </p>
              </div>
            </div>

            {/* Period Selector and Indicators */}
            <div className="mb-4 flex items-center justify-between gap-4">
              <ChartPeriodSelector selected={chartPeriod} onSelect={setChartPeriod} />
              
              {/* Technical Indicators Toggle */}
              <div className="relative" ref={indicatorMenuRef}>
                <button
                  onClick={() => setIndicatorMenuOpen(!indicatorMenuOpen)}
                  className={`
                    flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium
                    transition-all duration-200 border
                    ${enabledIndicators.size > 0 
                      ? 'bg-accent-cyan/20 border-accent-cyan/30 text-accent-cyan' 
                      : 'bg-white/5 border-white/10 text-white/70 hover:bg-white/10 hover:border-white/20'
                    }
                  `}
                >
                  <BarChart2 className="w-4 h-4" />
                  <span>Indicators</span>
                  {enabledIndicators.size > 0 && (
                    <span className="bg-accent-cyan/30 text-accent-cyan text-xs px-1.5 py-0.5 rounded-full">
                      {enabledIndicators.size}
                    </span>
                  )}
                  <ChevronDown className={`w-4 h-4 transition-transform ${indicatorMenuOpen ? 'rotate-180' : ''}`} />
                </button>
                
                {/* Indicator Dropdown Menu */}
                {indicatorMenuOpen && (
                  <div className="absolute right-0 top-full mt-2 w-64 bg-[rgba(15,15,20,0.98)] border border-white/10 rounded-xl shadow-2xl z-50 overflow-hidden fade-in">
                    {/* Moving Averages Group */}
                    <div className="p-2 border-b border-white/5">
                      <button
                        onClick={() => toggleIndicatorGroup('moving_averages')}
                        className="w-full flex items-center justify-between px-2 py-1.5 text-xs font-semibold text-white/50 uppercase tracking-wider hover:text-white/70"
                      >
                        <span>Moving Averages</span>
                        <span className="text-[10px] normal-case font-normal">Toggle All</span>
                      </button>
                      {INDICATOR_CONFIGS.filter(c => c.group === 'moving_averages').map(config => (
                        <IndicatorToggleItem
                          key={config.key}
                          config={config}
                          enabled={enabledIndicators.has(config.key)}
                          available={availableIndicators.some(a => a.key === config.key)}
                          onToggle={() => toggleIndicator(config.key)}
                        />
                      ))}
                    </div>
                    
                    {/* Bollinger Bands Group */}
                    <div className="p-2 border-b border-white/5">
                      <button
                        onClick={() => toggleIndicatorGroup('bollinger')}
                        className="w-full flex items-center justify-between px-2 py-1.5 text-xs font-semibold text-white/50 uppercase tracking-wider hover:text-white/70"
                      >
                        <span>Bollinger Bands</span>
                        <span className="text-[10px] normal-case font-normal">Toggle All</span>
                      </button>
                      {INDICATOR_CONFIGS.filter(c => c.group === 'bollinger').map(config => (
                        <IndicatorToggleItem
                          key={config.key}
                          config={config}
                          enabled={enabledIndicators.has(config.key)}
                          available={availableIndicators.some(a => a.key === config.key)}
                          onToggle={() => toggleIndicator(config.key)}
                        />
                      ))}
                    </div>
                    
                    {/* Volume-based Group */}
                    <div className="p-2">
                      <button
                        onClick={() => toggleIndicatorGroup('volume')}
                        className="w-full flex items-center justify-between px-2 py-1.5 text-xs font-semibold text-white/50 uppercase tracking-wider hover:text-white/70"
                      >
                        <span>Volume-Based</span>
                        <span className="text-[10px] normal-case font-normal">Toggle All</span>
                      </button>
                      {INDICATOR_CONFIGS.filter(c => c.group === 'volume').map(config => (
                        <IndicatorToggleItem
                          key={config.key}
                          config={config}
                          enabled={enabledIndicators.has(config.key)}
                          available={availableIndicators.some(a => a.key === config.key)}
                          onToggle={() => toggleIndicator(config.key)}
                        />
                      ))}
                    </div>
                    
                    {/* Clear All Button */}
                    {enabledIndicators.size > 0 && (
                      <div className="p-2 border-t border-white/5">
                        <button
                          onClick={() => setEnabledIndicators(new Set())}
                          className="w-full px-3 py-1.5 text-xs text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                        >
                          Clear All Indicators
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>
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
            <div ref={chartContainerRef} className="h-80 bg-white/5 rounded-xl p-4 border border-white/10 relative select-none">
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
                <ComposedChart 
                  data={processedHistory}
                  onMouseDown={handleChartMouseDown}
                  onMouseMove={handleChartMouseMove}
                  onMouseUp={handleChartMouseUp}
                  onMouseLeave={handleChartMouseLeave}
                  style={{ cursor: isSelecting ? 'crosshair' : 'crosshair' }}
                >
                  <defs>
                    <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={lineColor} stopOpacity={0.3}/>
                      <stop offset="95%" stopColor={lineColor} stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="extendedHoursGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#06b6d4" stopOpacity={0.2}/>
                      <stop offset="100%" stopColor="#10b981" stopOpacity={0.05}/>
                    </linearGradient>
                    <linearGradient id="selectionGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#6366f1" stopOpacity={0.4}/>
                      <stop offset="100%" stopColor="#6366f1" stopOpacity={0.1}/>
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
                      ticks={dailyAxisTicks}
                      interval={dailyAxisTicks ? 0 : "preserveStartEnd"}
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
                  
                  {/* User selection range */}
                  {selectionStart !== null && selectionEnd !== null && selectionStart !== selectionEnd && (
                    <ReferenceArea
                      x1={Math.min(selectionStart, selectionEnd)}
                      x2={Math.max(selectionStart, selectionEnd)}
                      fill="url(#selectionGradient)"
                      stroke="rgba(99, 102, 241, 0.6)"
                      strokeWidth={1}
                    />
                  )}
                  
                  {/* Reference Line - Previous Close */}
                  {referenceClose !== null && (
                    <ReferenceLine 
                      y={referenceClose} 
                      stroke="rgba(255,255,255,0.4)" 
                      strokeDasharray="4 4"
                      strokeWidth={1.5}
                    />
                  )}
                  
                  {/* Technical Indicator Lines */}
                  {INDICATOR_CONFIGS.map(config => {
                    if (!enabledIndicators.has(config.key)) return null;
                    // Check if any data point has this indicator value
                    const hasData = processedHistory.some((p: any) => p[config.key] !== null);
                    if (!hasData) return null;
                    
                    return (
                      <Line
                        key={config.key}
                        type="monotone"
                        dataKey={config.key}
                        stroke={config.color}
                        strokeWidth={1.5}
                        strokeDasharray={config.strokeDasharray}
                        dot={false}
                        activeDot={false}
                        isAnimationActive={false}
                        connectNulls={true}
                      />
                    );
                  })}
                  
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
                      
                      if (!isLastPoint && !showAllDots) return <g key={`empty-${index}`} />;
                      
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
              
              {/* Selection Statistics Tooltip */}
              {selectionStats && tooltipPosition && !isSelecting && (
                <SelectionTooltip
                  stats={selectionStats}
                  position={tooltipPosition}
                  onClose={clearSelection}
                  formatDate={formatDate}
                />
              )}
            </div>

            {/* Session Changes Display */}
            {sessionChanges && (sessionChanges.preMarket || sessionChanges.regular || sessionChanges.afterHours) && (
              <div className="mt-4 grid grid-cols-3 gap-2">
                <SessionBadge 
                  label="Pre-Market" 
                  icon={<Sunrise className="w-3.5 h-3.5" />}
                  change={sessionChanges.preMarket} 
                />
                <SessionBadge 
                  label="Regular" 
                  icon={<Sun className="w-3.5 h-3.5" />}
                  change={sessionChanges.regular} 
                />
                <SessionBadge 
                  label="After Hours" 
                  icon={<Moon className="w-3.5 h-3.5" />}
                  change={sessionChanges.afterHours} 
                />
              </div>
            )}

            {/* Legend */}
            <div className="mt-3 flex flex-wrap gap-x-4 gap-y-2 text-xs text-white/50">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-white/40" style={{ borderStyle: 'dashed' }}></span>
                Previous Close
              </span>
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
              {/* Enabled Indicator Legend Items */}
              {INDICATOR_CONFIGS.filter(c => enabledIndicators.has(c.key)).map(config => (
                <span key={config.key} className="flex items-center gap-1.5">
                  <span 
                    className="w-3 h-0.5" 
                    style={{ 
                      backgroundColor: config.color,
                      borderStyle: config.strokeDasharray ? 'dashed' : 'solid'
                    }}
                  ></span>
                  {config.label}
                </span>
              ))}
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
      <div className="flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sm bg-white/5 text-white/30 border border-white/5">
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
        flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sm font-medium
        ${isPositive 
          ? 'bg-green-500/15 text-green-400 border border-green-500/20' 
          : 'bg-red-500/15 text-red-400 border border-red-500/20'
        }
      `}
      title={`${label}: ${sign}${change.percent.toFixed(2)}% (${formatValue(change.value)})`}
    >
      <span className="opacity-70">{icon}</span>
      <span className="opacity-70">{label}</span>
      <span className="font-semibold">{sign}{change.percent.toFixed(2)}%</span>
      <span className="opacity-60 text-xs">{formatValue(change.value)}</span>
    </div>
  );
}

// Selection tooltip component - displays statistics for the selected range
interface SelectionStats {
  startDate: string;
  endDate: string;
  startPrice: number;
  endPrice: number;
  priceChange: number;
  percentChange: number;
  rangeHigh: number;
  rangeLow: number;
  totalVolume: number;
  numPoints: number;
}

function SelectionTooltip({
  stats,
  position,
  onClose,
  formatDate
}: {
  stats: SelectionStats;
  position: { x: number; y: number };
  onClose: () => void;
  formatDate: (dateStr: string, includeTime: boolean) => string;
}) {
  const isPositive = stats.percentChange >= 0;
  const sign = isPositive ? '+' : '';
  
  // Format volume with K/M suffix
  const formatVolume = (vol: number): string => {
    if (vol >= 1_000_000) return `${(vol / 1_000_000).toFixed(1)}M`;
    if (vol >= 1_000) return `${(vol / 1_000).toFixed(1)}K`;
    return vol.toString();
  };
  
  return (
    <div
      className="absolute z-20 bg-[rgba(10,10,20,0.95)] border border-indigo-500/30 rounded-lg shadow-xl backdrop-blur-sm fade-in"
      style={{
        left: position.x,
        top: position.y,
        minWidth: '200px'
      }}
    >
      {/* Header with close button */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-white/10">
        <span className="text-xs text-indigo-400 font-medium">Selection Range</span>
        <button
          onClick={onClose}
          className="p-0.5 rounded hover:bg-white/10 transition-colors"
        >
          <X className="w-3.5 h-3.5 text-white/50 hover:text-white/80" />
        </button>
      </div>
      
      {/* Main price change */}
      <div className="px-3 py-2 border-b border-white/5">
        <div className="flex items-center gap-2">
          {isPositive ? (
            <ArrowUpRight className="w-5 h-5 text-green-400" />
          ) : (
            <ArrowDownRight className="w-5 h-5 text-red-400" />
          )}
          <span className={`text-xl font-bold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
            {sign}{stats.percentChange.toFixed(2)}%
          </span>
        </div>
        <p className={`text-sm mt-0.5 ${isPositive ? 'text-green-400/70' : 'text-red-400/70'}`}>
          {sign}${stats.priceChange.toFixed(2)}
        </p>
      </div>
      
      {/* Details grid */}
      <div className="px-3 py-2 space-y-1.5 text-xs">
        {/* Time range */}
        <div className="flex items-center gap-2 text-white/60">
          <Clock className="w-3 h-3" />
          <span className="truncate">{formatDate(stats.startDate, true)}</span>
        </div>
        <div className="flex items-center gap-2 text-white/60 pl-5">
          <span className="text-white/30">→</span>
          <span className="truncate">{formatDate(stats.endDate, true)}</span>
        </div>
        
        {/* Price range */}
        <div className="flex justify-between mt-2 pt-2 border-t border-white/5">
          <span className="text-white/40">Start</span>
          <span className="text-white/80 font-medium">${stats.startPrice.toFixed(2)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-white/40">End</span>
          <span className="text-white/80 font-medium">${stats.endPrice.toFixed(2)}</span>
        </div>
        
        {/* High/Low */}
        <div className="flex justify-between mt-2 pt-2 border-t border-white/5">
          <span className="text-white/40">Range High</span>
          <span className="text-green-400/80">${stats.rangeHigh.toFixed(2)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-white/40">Range Low</span>
          <span className="text-red-400/80">${stats.rangeLow.toFixed(2)}</span>
        </div>
        
        {/* Volume */}
        {stats.totalVolume > 0 && (
          <div className="flex justify-between mt-2 pt-2 border-t border-white/5">
            <span className="text-white/40">Volume</span>
            <span className="text-white/80">{formatVolume(stats.totalVolume)}</span>
          </div>
        )}
        
        {/* Data points count */}
        <div className="flex justify-between">
          <span className="text-white/40">Data Points</span>
          <span className="text-white/60">{stats.numPoints}</span>
        </div>
      </div>
    </div>
  );
}

// Indicator toggle menu item component
function IndicatorToggleItem({
  config,
  enabled,
  available,
  onToggle,
}: {
  config: IndicatorConfig;
  enabled: boolean;
  available: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      onClick={onToggle}
      disabled={!available}
      className={`
        w-full flex items-center gap-3 px-2 py-2 rounded-lg text-sm transition-colors
        ${!available 
          ? 'opacity-40 cursor-not-allowed' 
          : enabled 
            ? 'bg-white/10 text-white' 
            : 'text-white/70 hover:bg-white/5 hover:text-white'
        }
      `}
      title={!available ? `Requires at least ${config.minPeriod} data points` : undefined}
    >
      {/* Color indicator */}
      <span 
        className="w-4 h-0.5 flex-shrink-0 rounded-full"
        style={{ 
          backgroundColor: config.color,
          borderStyle: config.strokeDasharray ? 'dashed' : 'solid'
        }}
      />
      
      {/* Label */}
      <span className="flex-1 text-left">{config.label}</span>
      
      {/* Check indicator */}
      {enabled && (
        <Check className="w-4 h-4 text-accent-cyan" />
      )}
      
      {/* Unavailable indicator */}
      {!available && (
        <span className="text-[10px] text-white/30">
          Need {config.minPeriod}+ pts
        </span>
      )}
    </button>
  );
}
