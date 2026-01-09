"""
Golden Master Parity Test

Verifies that Python indicator calculations produce identical results
to the TypeScript frontend. This is CRITICAL for WFO calibration accuracy.

Without parity, the optimizer calibrates weights for a "ghost strategy"
that differs from the live trading system.

Usage:
1. Export golden_master.json from the frontend using goldenMasterExport.ts
2. Place it in backend/tests/fixtures/golden_master.json
3. Run: pytest backend/tests/test_indicator_parity.py -v

The test will:
- Load the golden master data (OHLCV + expected signals)
- Calculate signals using Python indicators
- Assert parity to 4 decimal places (after warmup period)
"""

import json
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Import the Python indicator calculator
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.indicators import (
    calculate_all_indicators,
    calculate_indicator_signals,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_adx,
    calculate_cmf,
    calculate_momentum,
    calculate_volume_analysis,
    calculate_rvol,
    calculate_sma_alignment,
    calculate_price_position,
)


# ============================================================================
# Configuration
# ============================================================================

# Warmup period: Ignore mismatches in first N rows due to EMA convergence
# EMAs have an "infinite tail" - different history lengths cause different values
WARMUP_PERIOD = 50

# Tolerance for floating-point comparison (4 decimal places)
# STRICT: Required for accurate WFO calibration
# A 2% difference in signal = optimizing a ghost strategy
# 
# Note: 0.00015 allows for half-unit rounding differences at 4th decimal
# (Python and JavaScript may round edge cases differently)
TOLERANCE = 0.00015

# Path to golden master fixture
FIXTURES_DIR = Path(__file__).parent / "fixtures"
GOLDEN_MASTER_PATH = FIXTURES_DIR / "golden_master.json"


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def golden_master() -> Dict[str, Any]:
    """Load golden master data from JSON file."""
    if not GOLDEN_MASTER_PATH.exists():
        pytest.skip(
            f"Golden master file not found: {GOLDEN_MASTER_PATH}\n"
            "Generate it by running exportGoldenMaster() in the frontend."
        )
    
    with open(GOLDEN_MASTER_PATH, "r") as f:
        return json.load(f)


@pytest.fixture
def price_dataframe(golden_master: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert golden master price history to pandas DataFrame for indicator calculation.
    
    Uses the FULL price history (including warmup) so Python calculates
    indicators on the same data as TypeScript.
    """
    # Use full price history if available, otherwise fall back to dataPoints
    if "priceHistory" in golden_master:
        price_data = golden_master["priceHistory"]
    else:
        price_data = golden_master["dataPoints"]
    
    df = pd.DataFrame([
        {
            "date": dp["date"],
            "open": dp["open"],
            "high": dp["high"],
            "low": dp["low"],
            "close": dp["close"],
            "volume": dp["volume"],
        }
        for dp in price_data
    ])
    
    return df


@pytest.fixture
def expected_signals(golden_master: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract expected signals from golden master.
    
    The index is aligned to match the price_dataframe:
    - If priceHistory exists, signals start at index WARMUP_PERIOD
    - If not, signals start at index 0
    """
    data_points = golden_master["dataPoints"]
    
    signals = pd.DataFrame([dp["signals"] for dp in data_points])
    signals["date"] = [dp["date"] for dp in data_points]
    
    # Align index to match price_dataframe if priceHistory exists
    if "priceHistory" in golden_master:
        warmup = golden_master.get("warmupPeriod", WARMUP_PERIOD)
        signals.index = range(warmup, warmup + len(signals))
    
    return signals


# ============================================================================
# Helper Functions
# ============================================================================

def assert_series_equal(
    python_series: pd.Series,
    expected_series: pd.Series,
    signal_name: str,
    warmup: int = WARMUP_PERIOD,
    tolerance: float = TOLERANCE
) -> int:
    """
    Assert that two series are equal within tolerance, ignoring warmup period.
    
    Returns number of comparisons made.
    """
    # Skip warmup period
    py_vals = python_series.iloc[warmup:].values
    exp_vals = expected_series.iloc[warmup:].values
    
    # Handle null values
    py_nulls = pd.isna(py_vals)
    exp_nulls = pd.isna(exp_vals)
    
    # Both should be null in the same places
    null_mismatch = np.sum(py_nulls != exp_nulls)
    if null_mismatch > 0:
        # Find first mismatch for debugging
        mismatch_idx = np.where(py_nulls != exp_nulls)[0][0]
        pytest.fail(
            f"{signal_name}: Null mismatch at index {warmup + mismatch_idx}. "
            f"Python: {py_vals[mismatch_idx]} ({py_nulls[mismatch_idx]}), "
            f"Expected: {exp_vals[mismatch_idx]} ({exp_nulls[mismatch_idx]})"
        )
    
    # Compare non-null values
    valid_mask = ~py_nulls & ~exp_nulls
    if not np.any(valid_mask):
        # All values are null - no comparison needed
        return 0
    
    py_valid = py_vals[valid_mask].astype(float)
    exp_valid = exp_vals[valid_mask].astype(float)
    
    # Check absolute difference
    diff = np.abs(py_valid - exp_valid)
    max_diff = np.max(diff)
    
    if max_diff > tolerance:
        # Find the worst mismatch for debugging
        worst_idx = np.argmax(diff)
        valid_indices = np.where(valid_mask)[0]
        actual_idx = warmup + valid_indices[worst_idx]
        
        pytest.fail(
            f"{signal_name}: Mismatch at index {actual_idx}. "
            f"Python: {py_valid[worst_idx]:.6f}, "
            f"Expected: {exp_valid[worst_idx]:.6f}, "
            f"Diff: {diff[worst_idx]:.6f} > tolerance {tolerance}"
        )
    
    return len(py_valid)


# ============================================================================
# Test Cases
# ============================================================================

class TestIndicatorParity:
    """Test Python vs TypeScript indicator calculation parity."""
    
    def test_golden_master_loaded(self, golden_master):
        """Verify golden master file is loaded correctly."""
        assert "ticker" in golden_master
        assert "dataPoints" in golden_master
        assert len(golden_master["dataPoints"]) > WARMUP_PERIOD
        
        print(f"\n📊 Golden Master: {golden_master['ticker']}")
        print(f"   Date Range: {golden_master['dataRange']['start']} to {golden_master['dataRange']['end']}")
        print(f"   Total Days: {golden_master['dataRange']['totalDays']}")
        print(f"   Warmup: {golden_master['warmupPeriod']} days")
    
    def test_rsi_parity(self, price_dataframe, expected_signals):
        """Test RSI calculation parity."""
        rsi_df = calculate_rsi(price_dataframe)
        
        # Compare RSI value
        comparisons = assert_series_equal(
            rsi_df["rsi_value"],
            expected_signals["rsi_value"],
            "RSI Value"
        )
        print(f"✅ RSI Value: {comparisons} values compared")
    
    def test_macd_parity(self, price_dataframe, expected_signals):
        """Test MACD calculation parity."""
        macd_df = calculate_macd(price_dataframe)
        
        # Compare MACD line
        comparisons = assert_series_equal(
            macd_df["macd_line"],
            expected_signals["macd_line"],
            "MACD Line"
        )
        print(f"✅ MACD Line: {comparisons} values compared")
        
        # Compare histogram
        comparisons = assert_series_equal(
            macd_df["macd_histogram"],
            expected_signals["macd_histogram"],
            "MACD Histogram"
        )
        print(f"✅ MACD Histogram: {comparisons} values compared")
    
    def test_bollinger_bands_parity(self, price_dataframe, expected_signals):
        """Test Bollinger Bands calculation parity."""
        bb_df = calculate_bollinger_bands(price_dataframe)
        
        # Compare %B
        comparisons = assert_series_equal(
            bb_df["bb_percent_b"],
            expected_signals["bb_percent_b"],
            "Bollinger %B"
        )
        print(f"✅ Bollinger %B: {comparisons} values compared")
        
        # Compare bandwidth
        comparisons = assert_series_equal(
            bb_df["bb_bandwidth"],
            expected_signals["bb_bandwidth"],
            "Bollinger Bandwidth"
        )
        print(f"✅ Bollinger Bandwidth: {comparisons} values compared")
    
    def test_adx_parity(self, price_dataframe, expected_signals):
        """Test ADX calculation parity."""
        adx_df = calculate_adx(price_dataframe)
        
        comparisons = assert_series_equal(
            adx_df["adx_value"],
            expected_signals["adx_value"],
            "ADX Value"
        )
        print(f"✅ ADX Value: {comparisons} values compared")
    
    def test_cmf_parity(self, price_dataframe, expected_signals):
        """Test CMF calculation parity."""
        cmf_df = calculate_cmf(price_dataframe)
        
        comparisons = assert_series_equal(
            cmf_df["cmf_value"],
            expected_signals["cmf_value"],
            "CMF Value"
        )
        print(f"✅ CMF Value: {comparisons} values compared")
    
    def test_momentum_parity(self, price_dataframe, expected_signals):
        """Test Momentum calculation parity."""
        mom_df = calculate_momentum(price_dataframe)
        
        comparisons = assert_series_equal(
            mom_df["momentum_short"],
            expected_signals["momentum_short"],
            "Momentum Short"
        )
        print(f"✅ Momentum Short: {comparisons} values compared")
    
    def test_volume_ratio_parity(self, price_dataframe, expected_signals):
        """Test Volume Ratio calculation parity."""
        vol_df = calculate_volume_analysis(price_dataframe)
        
        comparisons = assert_series_equal(
            vol_df["volume_ratio"],
            expected_signals["volume_ratio"],
            "Volume Ratio"
        )
        print(f"✅ Volume Ratio: {comparisons} values compared")
    
    def test_rvol_parity(self, price_dataframe, expected_signals):
        """Test RVOL calculation parity."""
        rvol_df = calculate_rvol(price_dataframe)
        
        comparisons = assert_series_equal(
            rvol_df["rvol_value"],
            expected_signals["rvol_value"],
            "RVOL Value"
        )
        print(f"✅ RVOL Value: {comparisons} values compared")
    
    def test_sma_parity(self, price_dataframe, expected_signals):
        """Test SMA calculation parity."""
        sma_df = calculate_sma_alignment(price_dataframe)
        
        # SMA 20
        comparisons = assert_series_equal(
            sma_df["sma_20"],
            expected_signals["sma_20"],
            "SMA 20"
        )
        print(f"✅ SMA 20: {comparisons} values compared")
        
        # SMA 50
        comparisons = assert_series_equal(
            sma_df["sma_50"],
            expected_signals["sma_50"],
            "SMA 50"
        )
        print(f"✅ SMA 50: {comparisons} values compared")
        
        # SMA 200 (may have many nulls if insufficient data)
        if expected_signals["sma_200"].notna().sum() > 10:
            comparisons = assert_series_equal(
                sma_df["sma_200"],
                expected_signals["sma_200"],
                "SMA 200"
            )
            print(f"✅ SMA 200: {comparisons} values compared")
        else:
            print("⚠️ SMA 200: Skipped (insufficient data in golden master)")
    
    def test_position_parity(self, price_dataframe, expected_signals):
        """Test Price Position calculation parity."""
        pos_df = calculate_price_position(price_dataframe)
        
        comparisons = assert_series_equal(
            pos_df["price_range_position"],
            expected_signals["range_position"],
            "Range Position"
        )
        print(f"✅ Range Position: {comparisons} values compared")


class TestSignalInterpretation:
    """Test signal interpretation parity (the signal values, not just raw indicators)."""
    
    def test_rsi_signal_parity(self, price_dataframe, expected_signals):
        """Test RSI signal interpretation parity."""
        signals_df = calculate_indicator_signals(price_dataframe)
        
        comparisons = assert_series_equal(
            signals_df["rsi_signal"],
            expected_signals["rsi_signal"],
            "RSI Signal"
        )
        print(f"✅ RSI Signal: {comparisons} values compared")
    
    def test_macd_signal_parity(self, price_dataframe, expected_signals):
        """Test MACD signal interpretation parity."""
        signals_df = calculate_indicator_signals(price_dataframe)
        
        comparisons = assert_series_equal(
            signals_df["macd_signal"],
            expected_signals["macd_signal"],
            "MACD Signal"
        )
        print(f"✅ MACD Signal: {comparisons} values compared")
    
    def test_bollinger_signal_parity(self, price_dataframe, expected_signals):
        """Test Bollinger signal interpretation parity."""
        signals_df = calculate_indicator_signals(price_dataframe)
        
        comparisons = assert_series_equal(
            signals_df["bollinger_signal"],
            expected_signals["bollinger_signal"],
            "Bollinger Signal"
        )
        print(f"✅ Bollinger Signal: {comparisons} values compared")
    
    def test_adx_signal_parity(self, price_dataframe, expected_signals):
        """Test ADX signal interpretation parity."""
        signals_df = calculate_indicator_signals(price_dataframe)
        
        comparisons = assert_series_equal(
            signals_df["adx_signal"],
            expected_signals["adx_signal"],
            "ADX Signal"
        )
        print(f"✅ ADX Signal: {comparisons} values compared")
    
    def test_cmf_signal_parity(self, price_dataframe, expected_signals):
        """Test CMF signal interpretation parity."""
        signals_df = calculate_indicator_signals(price_dataframe)
        
        comparisons = assert_series_equal(
            signals_df["cmf_signal"],
            expected_signals["cmf_signal"],
            "CMF Signal"
        )
        print(f"✅ CMF Signal: {comparisons} values compared")


class TestParityReport:
    """Generate a summary parity report."""
    
    def test_full_parity_report(self, price_dataframe, expected_signals, golden_master):
        """Run all parity checks and generate summary report."""
        results = []
        
        # Calculate Python signals
        try:
            signals_df = calculate_indicator_signals(price_dataframe)
        except Exception as e:
            pytest.fail(f"Failed to calculate Python signals: {e}")
        
        # Define checks
        checks = [
            ("RSI Value", "rsi_value", lambda: calculate_rsi(price_dataframe)["rsi_value"]),
            ("RSI Signal", "rsi_signal", lambda: signals_df["rsi_signal"]),
            ("MACD Line", "macd_line", lambda: calculate_macd(price_dataframe)["macd_line"]),
            ("MACD Signal", "macd_signal", lambda: signals_df["macd_signal"]),
            ("BB %B", "bb_percent_b", lambda: calculate_bollinger_bands(price_dataframe)["bb_percent_b"]),
            ("BB Signal", "bollinger_signal", lambda: signals_df["bollinger_signal"]),
            ("ADX Value", "adx_value", lambda: calculate_adx(price_dataframe)["adx_value"]),
            ("ADX Signal", "adx_signal", lambda: signals_df["adx_signal"]),
            ("CMF Value", "cmf_value", lambda: calculate_cmf(price_dataframe)["cmf_value"]),
            ("CMF Signal", "cmf_signal", lambda: signals_df["cmf_signal"]),
        ]
        
        print(f"\n{'='*60}")
        print(f"  Golden Master Parity Report: {golden_master['ticker']}")
        print(f"  Date Range: {golden_master['dataRange']['start']} to {golden_master['dataRange']['end']}")
        print(f"  Warmup Period: {WARMUP_PERIOD} days (ignored)")
        print(f"  Tolerance: {TOLERANCE}")
        print(f"{'='*60}\n")
        
        passed = 0
        failed = 0
        
        for name, col, getter in checks:
            try:
                py_series = getter()
                exp_series = expected_signals[col]
                
                # Calculate stats
                valid_py = py_series.iloc[WARMUP_PERIOD:].dropna()
                valid_exp = exp_series.iloc[WARMUP_PERIOD:].dropna()
                
                if len(valid_py) == 0 or len(valid_exp) == 0:
                    print(f"⚠️  {name}: No valid data to compare")
                    continue
                
                # Calculate max difference
                common_idx = valid_py.index.intersection(valid_exp.index)
                if len(common_idx) == 0:
                    print(f"⚠️  {name}: No overlapping indices")
                    continue
                
                diff = (valid_py.loc[common_idx] - valid_exp.loc[common_idx].values).abs()
                max_diff = diff.max()
                mean_diff = diff.mean()
                
                if max_diff <= TOLERANCE:
                    print(f"✅ {name}: PASS (max diff: {max_diff:.6f}, mean: {mean_diff:.6f})")
                    passed += 1
                else:
                    print(f"❌ {name}: FAIL (max diff: {max_diff:.6f} > {TOLERANCE})")
                    failed += 1
                    
            except Exception as e:
                print(f"❌ {name}: ERROR - {e}")
                failed += 1
        
        print(f"\n{'='*60}")
        print(f"  Summary: {passed} passed, {failed} failed")
        print(f"{'='*60}\n")
        
        if failed > 0:
            pytest.fail(f"{failed} parity checks failed")


# ============================================================================
# Run as Script
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

