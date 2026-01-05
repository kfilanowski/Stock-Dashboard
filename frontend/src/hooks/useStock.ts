import { useState, useEffect } from 'react';
import type { StockData } from '../types';
import * as api from '../services/api';

export function useStock(ticker: string | null) {
  const [stock, setStock] = useState<StockData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch stock');
        setStock(null);
      } finally {
        setLoading(false);
      }
    };

    fetchStock();
  }, [ticker]);

  return { stock, loading, error };
}

