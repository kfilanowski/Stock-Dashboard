export type ChartPeriod = '1d' | '3d' | '1w' | '1mo' | '3mo' | '6mo' | 'ytd' | '1y' | '2y';

interface ChartPeriodSelectorProps {
  selected: ChartPeriod;
  onSelect: (period: ChartPeriod) => void;
}

const periods: { value: ChartPeriod; label: string }[] = [
  { value: '1d', label: '1D' },
  { value: '3d', label: '3D' },
  { value: '1w', label: '1W' },
  { value: '1mo', label: '1M' },
  { value: '3mo', label: '3M' },
  { value: '6mo', label: '6M' },
  { value: 'ytd', label: 'YTD' },
  { value: '1y', label: '1Y' },
  { value: '2y', label: '2Y' },
];

export function ChartPeriodSelector({ selected, onSelect }: ChartPeriodSelectorProps) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {periods.map(({ value, label }) => (
        <button
          key={value}
          onClick={() => onSelect(value)}
          className={`px-3 py-1 rounded-lg text-xs font-medium transition-all duration-200 ${
            selected === value
              ? 'bg-gradient-to-r from-accent-cyan to-accent-purple text-white shadow-lg shadow-accent-cyan/20'
              : 'bg-white/5 text-white/60 border border-white/10 hover:bg-white/10 hover:text-white/80'
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

