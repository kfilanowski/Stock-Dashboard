import { useState, useMemo } from 'react';
import { TrendingUp, TrendingDown, Trash2, BarChart3, Edit2, Check, X, RefreshCw, Calendar, DollarSign } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer, Tooltip, ReferenceLine, YAxis, ReferenceArea } from 'recharts';
import type { Holding, HistoryPoint } from '../types';

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
}

// Small inline update indicator
function UpdateIndicator({ updating }: { updating: boolean }) {
  if (!updating) return null;
  return (
    <RefreshCw className="w-3 h-3 text-accent-cyan/60 animate-spin inline ml-1" />
  );
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
  isHistoryLoading = false
}: HoldingCardProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [newAllocation, setNewAllocation] = useState(holding.allocation_pct.toString());
  const [newAmount, setNewAmount] = useState('');
  const [activeInput, setActiveInput] = useState<'percent' | 'amount' | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  
  // Investment date/price editing
  const [isEditingInvestment, setIsEditingInvestment] = useState(false);
  const [investmentDate, setInvestmentDate] = useState(
    holding.investment_date ? holding.investment_date.split('T')[0] : ''
  );
  const [investmentPrice, setInvestmentPrice] = useState(
    holding.investment_price?.toString() ?? ''
  );
  const [savingInvestment, setSavingInvestment] = useState(false);

  const isPositive = (holding.ytd_return ?? 0) >= 0;
  const priceVsSMA = holding.price_vs_sma ?? 0;
  const isAboveSMA = priceVsSMA >= 0;

  // Calculate period gain (latest close vs reference close)
  const periodGain = useMemo(() => {
    if (!history.length || referenceClose === null || referenceClose === 0) return null;
    const latestClose = history[history.length - 1]?.close ?? 0;
    const gain = ((latestClose - referenceClose) / referenceClose) * 100;
    return gain;
  }, [history, referenceClose]);

  // Determine if current price is above or below reference close
  const isAboveReference = useMemo(() => {
    if (periodGain === null) return isPositive;
    return periodGain >= 0;
  }, [periodGain, isPositive]);

  // Line color based on performance vs reference
  const lineColor = isAboveReference ? '#22c55e' : '#ef4444'; // green-500 or red-500

  // Calculate Y-axis domain to include reference close
  const yAxisDomain = useMemo(() => {
    if (!history.length) return ['auto', 'auto'] as const;
    
    const closes = history.map(h => h.close).filter(c => c > 0);
    if (!closes.length) return ['auto', 'auto'] as const;
    
    let min = Math.min(...closes);
    let max = Math.max(...closes);
    
    // Include reference close in the domain
    if (referenceClose !== null && referenceClose > 0) {
      min = Math.min(min, referenceClose);
      max = Math.max(max, referenceClose);
    }
    
    // Add 5% padding
    const padding = (max - min) * 0.05;
    return [min - padding, max + padding] as [number, number];
  }, [history, referenceClose]);

  // Process data to add extended hours info and find extended hours ranges
  const { processedHistory, extendedHoursRanges } = useMemo(() => {
    if (!history.length) return { processedHistory: [], extendedHoursRanges: [] };
    
    const processed = history.map((h, idx) => ({
      ...h,
      idx,
      isExtendedHours: !isMarketHours(h.date)
    }));
    
    // Find contiguous extended hours ranges for shading
    const ranges: { start: number; end: number }[] = [];
    let rangeStart: number | null = null;
    
    for (let i = 0; i < processed.length; i++) {
      if (processed[i].isExtendedHours) {
        if (rangeStart === null) rangeStart = i;
      } else {
        if (rangeStart !== null) {
          ranges.push({ start: rangeStart, end: i - 1 });
          rangeStart = null;
        }
      }
    }
    // Close any open range
    if (rangeStart !== null) {
      ranges.push({ start: rangeStart, end: processed.length - 1 });
    }
    
    return { processedHistory: processed, extendedHoursRanges: ranges };
  }, [history]);

  const maxAllocation = 100 - currentTotalAllocation + holding.allocation_pct;
  const maxAmount = portfolioTotalValue * (maxAllocation / 100);

  // Sync allocation and amount when editing
  const handleAllocationChange = (value: string) => {
    setActiveInput('percent');
    setNewAllocation(value);
    if (value !== '' && portfolioTotalValue > 0) {
      const pct = parseFloat(value);
      if (!isNaN(pct)) {
        const calculatedAmount = (pct / 100) * portfolioTotalValue;
        setNewAmount(calculatedAmount.toFixed(2));
      }
    } else {
      setNewAmount('');
    }
  };

  const handleAmountChange = (value: string) => {
    setActiveInput('amount');
    setNewAmount(value);
    if (value !== '' && portfolioTotalValue > 0) {
      const amt = parseFloat(value);
      if (!isNaN(amt)) {
        const calculatedPct = (amt / portfolioTotalValue) * 100;
        setNewAllocation(calculatedPct.toFixed(2));
      }
    } else {
      setNewAllocation('');
    }
  };

  // Convert 24-hour time (HH:MM) to 12-hour format
  const formatTime12Hour = (time24: string): string => {
    const [hours, minutes] = time24.split(':').map(Number);
    if (isNaN(hours) || isNaN(minutes)) return time24;
    const ampm = hours >= 12 ? 'PM' : 'AM';
    const hours12 = hours % 12 || 12;
    return `${hours12}:${minutes.toString().padStart(2, '0')} ${ampm}`;
  };

  // Format date/time for tooltip
  const formatTooltipDate = (dateStr: string): string => {
    if (!dateStr) return '';
    
    // Check if it contains time (format: "YYYY-MM-DD HH:MM")
    if (dateStr.includes(' ')) {
      const [datePart, timePart] = dateStr.split(' ');
      try {
        const date = new Date(datePart);
        if (isNaN(date.getTime())) return dateStr;
        const formattedDate = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        const time12 = formatTime12Hour(timePart);
        return `${formattedDate} — ${time12}`;
      } catch {
        return dateStr;
      }
    }
    
    // Daily format (YYYY-MM-DD)
    try {
      const date = new Date(dateStr);
      if (isNaN(date.getTime())) return dateStr;
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    } catch {
      return dateStr;
    }
  };

  // Calculate gain % for a specific price point
  const calculateGainFromRef = (price: number): string | null => {
    if (referenceClose === null || referenceClose === 0) return null;
    const gain = ((price - referenceClose) / referenceClose) * 100;
    const sign = gain >= 0 ? '+' : '';
    return `${sign}${gain.toFixed(2)}%`;
  };

  const handleSave = async () => {
    setError('');
    const value = parseFloat(newAllocation) || 0;
    
    if (value < 0) {
      setError('Cannot be negative');
      return;
    }
    
    if (value > maxAllocation) {
      setError(`Max: ${maxAllocation.toFixed(1)}%`);
      return;
    }

    setSaving(true);
    try {
      await onUpdateAllocation(holding.id, value);
      setIsEditing(false);
      setActiveInput(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update');
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    setNewAllocation(holding.allocation_pct.toString());
    setNewAmount(((holding.allocation_pct / 100) * portfolioTotalValue).toFixed(2));
    setActiveInput(null);
    setError('');
    setIsEditing(false);
  };

  // Initialize amount when starting to edit
  const handleStartEditing = () => {
    setNewAllocation(holding.allocation_pct.toString());
    setNewAmount(((holding.allocation_pct / 100) * portfolioTotalValue).toFixed(2));
    setActiveInput(null);
    setIsEditing(true);
  };

  const handleSaveInvestment = async () => {
    setSavingInvestment(true);
    try {
      const data: { investment_date?: string; investment_price?: number } = {};
      
      if (investmentDate) {
        data.investment_date = new Date(investmentDate).toISOString();
      }
      if (investmentPrice) {
        const price = parseFloat(investmentPrice);
        if (!isNaN(price) && price > 0) {
          data.investment_price = price;
        }
      }
      
      await onUpdateInvestment(holding.id, data);
      setIsEditingInvestment(false);
    } catch (err) {
      console.error('Failed to save investment info:', err);
    } finally {
      setSavingInvestment(false);
    }
  };

  const handleCancelInvestment = () => {
    setInvestmentDate(holding.investment_date ? holding.investment_date.split('T')[0] : '');
    setInvestmentPrice(holding.investment_price?.toString() ?? '');
    setIsEditingInvestment(false);
  };

  // Format investment date for display
  const formatInvestmentDate = (dateStr: string | null | undefined): string => {
    if (!dateStr) return '';
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    } catch {
      return '';
    }
  };

  // Custom tooltip with percentage gain
  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null;
    
    const price = payload[0]?.value;
    const gainStr = calculateGainFromRef(price);
    // Get the date from the data point itself, not from the label (which is unreliable without XAxis)
    const dateStr = payload[0]?.payload?.date ?? '';
    
    return (
      <div className="bg-[rgba(10,10,15,0.95)] border border-white/10 rounded-lg px-3 py-2 text-xs">
        <p className="text-white/50 mb-1">{formatTooltipDate(dateStr)}</p>
        <p className="text-white font-medium">${price?.toFixed(2)}</p>
        {gainStr && (
          <p className={`text-xs mt-0.5 ${price >= (referenceClose ?? 0) ? 'text-green-400' : 'text-red-400'}`}>
            {gainStr} vs prev close
          </p>
        )}
      </div>
    );
  };

  return (
    <div className="glass-card p-5 hover:border-white/20 transition-all duration-300 group">
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
            {isEditing ? (
              <div className="flex flex-col gap-1.5 mt-1">
                <div className="flex items-center gap-1.5">
                  <input
                    type="number"
                    value={newAllocation}
                    onChange={(e) => handleAllocationChange(e.target.value)}
                    className="w-32 px-2 py-1 text-sm bg-white/10 border border-white/20 rounded focus:border-accent-cyan/50 focus:outline-none"
                    min="0"
                    max={maxAllocation}
                    step="0.1"
                    placeholder="0"
                    autoFocus={activeInput !== 'amount'}
                  />
                  <span className="text-white/50 text-xs">%</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-white/50 text-xs">$</span>
                  <input
                    type="number"
                    value={newAmount}
                    onChange={(e) => handleAmountChange(e.target.value)}
                    className="w-36 px-2 py-1 text-sm bg-white/10 border border-white/20 rounded focus:border-accent-cyan/50 focus:outline-none"
                    min="0"
                    max={maxAmount}
                    step="0.01"
                    placeholder="0.00"
                    autoFocus={activeInput === 'amount'}
                  />
                </div>
                <div className="flex items-center gap-1">
                  <button
                    onClick={handleSave}
                    disabled={saving}
                    className="px-2 py-1 rounded bg-green-500/20 hover:bg-green-500/30 text-green-400 text-xs flex items-center gap-1"
                  >
                    <Check className="w-3 h-3" />
                    Save
                  </button>
                  <button
                    onClick={handleCancel}
                    className="px-2 py-1 rounded bg-white/10 hover:bg-white/20 text-white/70 text-xs"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <p className="text-white/50 text-sm">{holding.allocation_pct}% allocation</p>
                <button
                  onClick={handleStartEditing}
                  className="p-1 rounded hover:bg-white/10 text-white/40 hover:text-white/70 transition-colors"
                  title="Edit allocation"
                >
                  <Edit2 className="w-3 h-3" />
                </button>
              </div>
            )}
            {error && <p className="text-red-400 text-xs mt-1">{error}</p>}
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

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <p className="text-white/50 text-xs mb-1">Current Price</p>
          <p className="text-white font-semibold">
            ${holding.current_price?.toFixed(2) ?? '—'}
            <UpdateIndicator updating={isRefreshing} />
          </p>
        </div>
        <div>
          <p className="text-white/50 text-xs mb-1">Value</p>
          <p className="text-white font-semibold">
            ${holding.current_value?.toLocaleString('en-US', { minimumFractionDigits: 2 }) ?? '—'}
            <UpdateIndicator updating={isRefreshing} />
          </p>
        </div>
        <div>
          <p className="text-white/50 text-xs mb-1">YTD Return</p>
          <div className="flex items-center gap-1">
            {isPositive ? (
              <TrendingUp className="w-4 h-4 text-green-400" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-400" />
            )}
            <p className={`font-semibold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
              {isPositive ? '+' : ''}{holding.ytd_return?.toFixed(2) ?? 0}%
            </p>
            <UpdateIndicator updating={isRefreshing} />
          </div>
        </div>
        <div>
          <p className="text-white/50 text-xs mb-1">Period Gain</p>
          {!isDataComplete ? (
            <p className="text-yellow-500/70 font-semibold text-xs" title="Incomplete historical data">
              Loading...
            </p>
          ) : periodGain !== null ? (
            <div className="flex items-center gap-1">
              {isAboveReference ? (
                <TrendingUp className="w-4 h-4 text-green-400" />
              ) : (
                <TrendingDown className="w-4 h-4 text-red-400" />
              )}
              <p className={`font-semibold ${isAboveReference ? 'text-green-400' : 'text-red-400'}`}>
                {periodGain >= 0 ? '+' : ''}{periodGain.toFixed(2)}%
              </p>
            </div>
          ) : (
            <p className="text-white/50 font-semibold">—</p>
          )}
        </div>
      </div>

      {/* Investment Info Section */}
      <div className="mb-3 p-2.5 rounded-lg bg-white/[0.03] border border-white/5">
        {isEditingInvestment ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Calendar className="w-3.5 h-3.5 text-white/40" />
              <input
                type="date"
                value={investmentDate}
                onChange={(e) => setInvestmentDate(e.target.value)}
                className="flex-1 px-2 py-1 text-xs bg-white/5 border border-white/20 rounded text-white focus:border-accent-cyan/50 focus:outline-none"
                max={new Date().toISOString().split('T')[0]}
              />
            </div>
            <div className="flex items-center gap-2">
              <DollarSign className="w-3.5 h-3.5 text-white/40" />
              <input
                type="number"
                value={investmentPrice}
                onChange={(e) => setInvestmentPrice(e.target.value)}
                placeholder="Price at investment"
                className="flex-1 px-2 py-1 text-xs bg-white/5 border border-white/20 rounded text-white focus:border-accent-cyan/50 focus:outline-none"
                min="0.01"
                step="0.01"
              />
              {holding.current_price && (
                <button
                  type="button"
                  onClick={() => setInvestmentPrice(holding.current_price!.toFixed(2))}
                  className="px-1.5 py-1 text-[10px] rounded bg-white/5 hover:bg-white/10 text-white/50 hover:text-white/70 whitespace-nowrap"
                  title="Use current price"
                >
                  Use ${holding.current_price.toFixed(2)}
                </button>
              )}
            </div>
            <div className="flex items-center gap-2 justify-end">
              <button
                onClick={handleSaveInvestment}
                disabled={savingInvestment}
                className="px-2 py-1 text-xs rounded bg-accent-cyan/20 hover:bg-accent-cyan/30 text-accent-cyan flex items-center gap-1"
              >
                <Check className="w-3 h-3" />
                Save
              </button>
              <button
                onClick={handleCancelInvestment}
                className="px-2 py-1 text-xs rounded bg-white/5 hover:bg-white/10 text-white/60"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 text-xs">
              {holding.investment_date ? (
                <>
                  <span className="text-white/40 flex items-center gap-1">
                    <Calendar className="w-3 h-3" />
                    {formatInvestmentDate(holding.investment_date)}
                  </span>
                  {holding.investment_price && (
                    <span className="text-white/40 flex items-center gap-1">
                      <DollarSign className="w-3 h-3" />
                      ${holding.investment_price.toFixed(2)}
                    </span>
                  )}
                  {holding.gain_loss_pct !== null && holding.gain_loss_pct !== undefined && (
                    <span className={`font-medium ${holding.gain_loss_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {holding.gain_loss_pct >= 0 ? '+' : ''}{holding.gain_loss_pct.toFixed(1)}%
                    </span>
                  )}
                </>
              ) : (
                <span className="text-white/30 italic">No investment date set</span>
              )}
            </div>
            <button
              onClick={() => setIsEditingInvestment(true)}
              className="p-1 rounded hover:bg-white/10 text-white/40 hover:text-white/70 transition-colors"
              title="Edit investment info"
            >
              <Edit2 className="w-3 h-3" />
            </button>
          </div>
        )}
      </div>

      {/* Mini Chart with Reference Line */}
      <div className="h-20 mt-2 relative">
        {isHistoryLoading && (
          <div className="absolute top-0 right-0 z-10">
            <RefreshCw className="w-3 h-3 text-accent-cyan/60 animate-spin" />
          </div>
        )}
        {/* Missing data indicator - gray left section */}
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
                <linearGradient id={`extendedHoursGradient-${holding.id}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#06b6d4" stopOpacity={0.15}/>
                  <stop offset="100%" stopColor="#10b981" stopOpacity={0.05}/>
                </linearGradient>
              </defs>
              <YAxis 
                domain={yAxisDomain} 
                hide 
              />
              <Tooltip content={<CustomTooltip />} />
              {/* Extended hours shading (pre-market & after-hours) */}
              {extendedHoursRanges.map((range, idx) => (
                <ReferenceArea
                  key={idx}
                  x1={range.start}
                  x2={range.end}
                  fill={`url(#extendedHoursGradient-${holding.id})`}
                  strokeOpacity={0}
                />
              ))}
              {/* Reference Line - Previous Close (gray dotted) */}
              {referenceClose !== null && (
                <ReferenceLine 
                  y={referenceClose} 
                  stroke="rgba(255,255,255,0.35)" 
                  strokeDasharray="3 3"
                  strokeWidth={1}
                />
              )}
              <Line
                type="monotone"
                dataKey="close"
                stroke={lineColor}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : !isHistoryLoading ? (
          <div className="w-full h-full flex items-center justify-center text-white/30 text-xs">
            No chart data
          </div>
        ) : null}
      </div>

      {/* Reference indicator */}
      {processedHistory.length > 0 && (
        <div className="mt-1 flex items-center gap-3 text-[10px] text-white/40">
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
          {!isDataComplete && (
            <span className="inline-flex items-center gap-1 text-yellow-500/70">
              <span className="w-3 h-2 rounded-sm bg-gradient-to-r from-gray-500/30 to-transparent border-l border-dashed border-gray-500/50"></span>
              Incomplete
            </span>
          )}
        </div>
      )}
    </div>
  );
}
