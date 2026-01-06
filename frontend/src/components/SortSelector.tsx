import { ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';

export type SortField = 
  | 'ticker' 
  | 'daily_change' 
  | 'allocation' 
  | 'equity' 
  | 'ytd'
  | 'confidence';

export type SortDirection = 'asc' | 'desc';

export interface SortOption {
  field: SortField;
  direction: SortDirection;
}

const SORT_OPTIONS: { field: SortField; label: string; defaultDirection: SortDirection }[] = [
  { field: 'ticker', label: 'Ticker (A-Z)', defaultDirection: 'asc' },
  { field: 'daily_change', label: 'Daily Change', defaultDirection: 'desc' },
  { field: 'allocation', label: 'Allocation %', defaultDirection: 'desc' },
  { field: 'equity', label: 'Equity Value', defaultDirection: 'desc' },
  { field: 'ytd', label: 'YTD Returns', defaultDirection: 'desc' },
  { field: 'confidence', label: 'Action Score', defaultDirection: 'desc' },
];

interface SortSelectorProps {
  value: SortOption;
  onChange: (option: SortOption) => void;
}

export function SortSelector({ value, onChange }: SortSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const currentOption = SORT_OPTIONS.find(opt => opt.field === value.field);
  
  const handleSelect = (field: SortField) => {
    const option = SORT_OPTIONS.find(opt => opt.field === field)!;
    
    // If same field, toggle direction; otherwise use default direction
    if (field === value.field) {
      onChange({ field, direction: value.direction === 'asc' ? 'desc' : 'asc' });
    } else {
      onChange({ field, direction: option.defaultDirection });
    }
    setIsOpen(false);
  };

  const toggleDirection = (e: React.MouseEvent) => {
    e.stopPropagation();
    onChange({ 
      field: value.field, 
      direction: value.direction === 'asc' ? 'desc' : 'asc' 
    });
  };

  const getDirectionLabel = () => {
    if (value.field === 'ticker') {
      return value.direction === 'asc' ? 'A → Z' : 'Z → A';
    }
    return value.direction === 'asc' ? 'Low → High' : 'High → Low';
  };

  return (
    <div className="relative flex items-center gap-1" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 hover:border-white/20 hover:bg-white/10 transition-all text-sm"
      >
        <ArrowUpDown className="w-4 h-4 text-white/50" />
        <span className="text-white/70">{currentOption?.label}</span>
      </button>
      <button
        onClick={toggleDirection}
        className="p-1.5 rounded-lg bg-white/5 border border-white/10 hover:border-white/20 hover:bg-white/10 transition-colors"
        title={`Sort ${value.direction === 'asc' ? 'ascending' : 'descending'}`}
      >
        {value.direction === 'asc' ? (
          <ArrowUp className="w-3.5 h-3.5 text-accent-cyan" />
        ) : (
          <ArrowDown className="w-3.5 h-3.5 text-accent-cyan" />
        )}
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 mt-1 w-48 py-1 rounded-lg bg-[#1a1a24] border border-white/10 shadow-xl z-50 animate-in fade-in slide-in-from-top-1 duration-150">
          <div className="px-3 py-1.5 text-xs text-white/40 uppercase tracking-wider">
            Sort by
          </div>
          {SORT_OPTIONS.map((option) => (
            <button
              key={option.field}
              onClick={() => handleSelect(option.field)}
              className={`w-full px-3 py-2 text-left text-sm hover:bg-white/5 transition-colors flex items-center justify-between ${
                value.field === option.field ? 'text-accent-cyan' : 'text-white/70'
              }`}
            >
              <span>{option.label}</span>
              {value.field === option.field && (
                <span className="text-xs text-white/40">
                  {getDirectionLabel()}
                </span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

