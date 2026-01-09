"""
Generate Golden Master Fixture

This script generates a golden_master.json file for parity testing.
For TRUE parity testing, this should be generated from the TypeScript frontend.
This Python version validates the test infrastructure works.

Usage:
    python backend/tests/generate_golden_master.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import pandas as pd
import numpy as np

from app.services.indicators import (
    calculate_all_indicators,
    calculate_indicator_signals,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_bollinger_squeeze,
    calculate_adx,
    calculate_cmf,
    calculate_momentum,
    calculate_volume_analysis,
    calculate_rvol,
    calculate_sma_alignment,
    calculate_price_position,
)


def fetch_stock_data(ticker: str, days: int = 500) -> pd.DataFrame:
    """Fetch historical stock data using yfinance."""
    print(f"Fetching {days} days of {ticker} data...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days * 1.5)  # Extra buffer for weekends
    
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    
    if len(df) < 100:
        raise ValueError(f"Insufficient data: got {len(df)} days, need at least 100")
    
    # Standardize column names
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Keep only required columns
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    print(f"Got {len(df)} days of data: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    return df


def generate_golden_master(ticker: str = "AAPL") -> dict:
    """Generate golden master data structure."""
    
    # Fetch data
    df = fetch_stock_data(ticker, days=400)
    
    # Calculate all indicators
    print("Calculating indicators...")
    indicators_df = calculate_all_indicators(df)
    signals_df = calculate_indicator_signals(df)
    
    # Build data points
    data_points = []
    warmup = 50
    
    for i in range(warmup, len(df)):
        row = df.iloc[i]
        ind_row = indicators_df.iloc[i]
        sig_row = signals_df.iloc[i]
        
        def safe_float(val, decimals=4):
            """Convert to float, handling NaN."""
            if pd.isna(val):
                return None
            return round(float(val), decimals)
        
        data_point = {
            "date": row['date'],
            "open": safe_float(row['open'], 2),
            "high": safe_float(row['high'], 2),
            "low": safe_float(row['low'], 2),
            "close": safe_float(row['close'], 2),
            "volume": int(row['volume']) if not pd.isna(row['volume']) else 0,
            "signals": {
                "rsi_signal": safe_float(sig_row.get('rsi_signal')),
                "rsi_value": safe_float(ind_row.get('rsi_value')),
                "macd_signal": safe_float(sig_row.get('macd_signal')),
                "macd_line": safe_float(ind_row.get('macd_line')),
                "macd_histogram": safe_float(ind_row.get('macd_histogram')),
                "bollinger_signal": safe_float(sig_row.get('bollinger_signal')),
                "bb_percent_b": safe_float(ind_row.get('bb_percent_b')),
                "bb_bandwidth": safe_float(ind_row.get('bb_bandwidth')),
                "adx_signal": safe_float(sig_row.get('adx_signal')),
                "adx_value": safe_float(ind_row.get('adx_value')),
                "cmf_signal": safe_float(sig_row.get('cmf_signal')),
                "cmf_value": safe_float(ind_row.get('cmf_value')),
                "momentum_signal": safe_float(sig_row.get('momentum_signal')),
                "momentum_short": safe_float(ind_row.get('momentum_short')),
                "volume_signal": safe_float(sig_row.get('volume_signal')),
                "volume_ratio": safe_float(ind_row.get('volume_ratio')),
                "rvol_signal": safe_float(sig_row.get('rvol_signal')),
                "rvol_value": safe_float(ind_row.get('rvol_value')),
                "sma_signal": safe_float(sig_row.get('sma_signal')),
                "sma_20": safe_float(ind_row.get('sma_20')),
                "sma_50": safe_float(ind_row.get('sma_50')),
                "sma_200": safe_float(ind_row.get('sma_200')),
                "position_signal": safe_float(sig_row.get('position_signal')),
                "range_position": safe_float(ind_row.get('price_range_position')),
                "squeeze_signal": safe_float(sig_row.get('squeeze_signal')),
                "squeeze_percentile": safe_float(ind_row.get('squeeze_percentile'), 0),
            }
        }
        data_points.append(data_point)
    
    # Calculate summary ranges
    rsi_vals = [dp['signals']['rsi_value'] for dp in data_points if dp['signals']['rsi_value'] is not None]
    macd_vals = [dp['signals']['macd_line'] for dp in data_points if dp['signals']['macd_line'] is not None]
    adx_vals = [dp['signals']['adx_value'] for dp in data_points if dp['signals']['adx_value'] is not None]
    cmf_vals = [dp['signals']['cmf_value'] for dp in data_points if dp['signals']['cmf_value'] is not None]
    
    golden_master = {
        "ticker": ticker,
        "exportedAt": datetime.now().isoformat(),
        "dataRange": {
            "start": data_points[0]['date'],
            "end": data_points[-1]['date'],
            "totalDays": len(data_points)
        },
        "warmupPeriod": warmup,
        "dataPoints": data_points,
        "summary": {
            "rsiRange": [round(min(rsi_vals), 2), round(max(rsi_vals), 2)] if rsi_vals else [0, 0],
            "macdRange": [round(min(macd_vals), 4), round(max(macd_vals), 4)] if macd_vals else [0, 0],
            "adxRange": [round(min(adx_vals), 2), round(max(adx_vals), 2)] if adx_vals else [0, 0],
            "cmfRange": [round(min(cmf_vals), 4), round(max(cmf_vals), 4)] if cmf_vals else [0, 0],
        },
        "_note": "Generated from Python for infrastructure validation. True parity requires TypeScript export."
    }
    
    return golden_master


def main():
    """Generate and save golden master fixture."""
    
    # Generate data
    golden_master = generate_golden_master("AAPL")
    
    # Save to fixtures
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)
    
    output_path = fixtures_dir / "golden_master.json"
    
    with open(output_path, "w") as f:
        json.dump(golden_master, f, indent=2)
    
    print(f"\n✅ Golden master saved to: {output_path}")
    print(f"   Ticker: {golden_master['ticker']}")
    print(f"   Date Range: {golden_master['dataRange']['start']} to {golden_master['dataRange']['end']}")
    print(f"   Total Days: {golden_master['dataRange']['totalDays']}")
    print(f"   RSI Range: {golden_master['summary']['rsiRange']}")
    print(f"   ADX Range: {golden_master['summary']['adxRange']}")


if __name__ == "__main__":
    main()

