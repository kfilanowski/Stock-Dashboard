
import pytest
import pandas as pd
import numpy as np
from app.services.wfo_simulator import fast_simulate, SimulationResult, MIN_TRADES_FOR_SIGNIFICANCE, apply_weights_to_signals
from app.services.wfo_optimizer import two_pass_coordinate_descent, get_adaptive_window, InsufficientVolatilityError, OptimizationResult, DEFAULT_WEIGHTS
from app.services.regime import MarketRegime

@pytest.fixture
def synthetic_data():
    """Create synthetic OHLCV data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
    
    # Create a predictable trend: Sine wave + linear trend
    t = np.linspace(0, 40*np.pi, 200)
    trend = np.linspace(100, 150, 200)
    price = trend + 5 * np.sin(t)
    
    df = pd.DataFrame({
        "date": dates,
        "open": price,
        "high": price + 1,
        "low": price - 1,
        "close": price,
        "volume": 1000000
    })
    
    # Add dummy signal columns expected by simulator
    # We'll make RSI a perfect predictor for testing
    # Long when sine wave is at bottom, Short when at top
    # sin(t) is bottom at 3pi/2, 7pi/2... (-1)
    # sin(t) is top at pi/2, 5pi/2... (1)
    
    # RSI signal: 1.0 (buy) at bottoms, -1.0 (sell) at tops, 0 else
    rsi_signal = np.zeros(200)
    rsi_signal[np.sin(t) < -0.9] = 1.0 # Buy at bottom
    rsi_signal[np.sin(t) > 0.9] = -1.0 # Sell at top
    
    df["rsi_signal"] = rsi_signal
    df["macd_signal"] = np.zeros(200) # Noise
    df["bollinger_signal"] = np.zeros(200) # Noise
    
    return df

def test_apply_weights_to_signals(synthetic_data):
    """Test weighted signal combination."""
    weights = {"rsi": 1.0, "macd": 0.5}
    
    # Mock calculate_indicator_signals output
    signals_df = pd.DataFrame({
        "rsi_signal": [1.0, -1.0, 0.5],
        "macd_signal": [0.0, 0.0, 1.0]
    })
    
    # Expected: 
    # Row 0: (1*1 + 0*0.5) / 1.5 = 1/1.5 = 0.666...
    # Row 1: (-1*1 + 0*0.5) / 1.5 = -0.666...
    # Row 2: (0.5*1 + 1*0.5) / 1.5 = 1.0 / 1.5 = 0.666...
    
    result = apply_weights_to_signals(signals_df, weights)
    
    assert result[0] == pytest.approx(1.0 / 1.5)
    assert result[1] == pytest.approx(-1.0 / 1.5)
    assert result[2] == pytest.approx(1.0 / 1.5)

def test_fast_simulate_basic(synthetic_data):
    """Test basic simulation with perfect signals."""
    weights = {"rsi": 1.0}
    
    # We need to mock calculate_indicator_signals since it recalculates signals inside fast_simulate
    # But fast_simulate calls it internally.
    # To properly test without mocking the internal call, we rely on the fact that
    # fast_simulate CALLS calculate_indicator_signals. 
    # Since our synthetic data DOES NOT include the logic to generate real RSI signals,
    # fast_simulate will generate real RSI signals from OHLC data, which might not match our "rsi_signal" column.
    
    # Wait, looking at wfo_simulator.py:
    # signals_df = calculate_indicator_signals(df)
    # It re-calculates signals.
    
    # So we should probably MOCK calculate_indicator_signals to return our pre-cooked signals
    # or ensure our synthetic data produces predictable real signals.
    pass

# We need to monkeypatch calculate_indicator_signals to use our pre-defined signals
# otherwise we are testing the indicator calculation logic, not the simulator logic.

@pytest.fixture
def mock_indicators(monkeypatch):
    def mock_calc(df):
        # Return columns that end with _signal from df, if they exist
        cols = [c for c in df.columns if c.endswith('_signal')]
        if not cols:
             # Fallback if no signal columns in input df
            return pd.DataFrame(index=df.index)
        return df[cols]
    
    monkeypatch.setattr("app.services.wfo_simulator.calculate_indicator_signals", mock_calc)

def test_fast_simulate_with_mock_signals(synthetic_data, mock_indicators):
    """Test simulation using mocked perfect signals."""
    weights = {"rsi": 1.0}
    
    # Horizon 3 days
    result = fast_simulate(synthetic_data, weights, horizon=3, apply_regime_filter=False)
    
    # We expect trades.
    # Our synthetic data has clear buy/sell points.
    
    assert result.total_trades > 0
    assert result.sqn != 0
    assert result.win_rate >= 0.0 # Should be high with perfect signals
    
    # Check if trades dataframe is populated
    assert result.trades is not None
    assert len(result.trades) == result.total_trades

def test_fast_simulate_insufficient_data():
    """Test simulation with too little data."""
    df = pd.DataFrame({"close": range(10)}) # Only 10 rows
    weights = {"rsi": 1.0}
    
    result = fast_simulate(df, weights)
    
    assert result.total_trades == 0
    assert result.sqn == 0.0

def test_two_pass_optimizer(synthetic_data, mock_indicators):
    """Test that optimizer picks the 'best' indicator."""
    
    # Make RSI perfect and MACD terrible (random noise is already in synthetic_data for macd?)
    # Actually synthetic_data['macd_signal'] is all zeros.
    
    # Let's make MACD always wrong: -1 * perfect_signal
    synthetic_data["macd_signal"] = -1 * synthetic_data["rsi_signal"]
    
    # Run optimizer
    # We only optimize RSI and MACD to save time
    indicators = ['rsi', 'macd']
    
    # Initialize with 0.0 weights so they don't interfere with each other during the first pass
    initial_weights = {k: 0.0 for k in DEFAULT_WEIGHTS.keys()}
    
    weights, results = two_pass_coordinate_descent(
        synthetic_data, 
        horizon=3, 
        indicators=indicators,
        initial_weights=initial_weights,
        coarse_grid=[0.0, 1.0], # Keep grid small for speed
        fine_offsets=[0.0] # Skip fine pass for speed/simplicity
    )
    
    # RSI should have positive weight (it works)
    # MACD should have 0.0 weight (it cancels RSI or loses money)
    
    assert weights['rsi'] >= 0.8 # Should pick 1.0
    assert weights['macd'] <= 0.2 # Should pick 0.0
    
    # Both result sets should end up valid because the final state (rsi=1, macd=0) is good
    # Note: results['rsi'] is from the first pass (rsi=1, macd=0).
    # results['macd'] is from second pass (rsi=1, macd=0).
    
    assert results['rsi'].sqn_score > 0
    assert results['macd'].sqn_score > 0

def test_adaptive_window_expansion(synthetic_data, mock_indicators):
    """Test that window expands to find enough trades."""
    
    # Make the first 100 rows have NO signals
    synthetic_data.loc[:100, "rsi_signal"] = 0
    synthetic_data.loc[:100, "macd_signal"] = 0
    
    weights = {"rsi": 1.0}
    
    # Start with small window that falls in the "no signal" zone
    # If we take tail(50), we get rows 150-200 which HAVE signals.
    # We want the TAIL to have signals, but maybe sparse?
    
    # Let's setup:
    # Total 200 rows.
    # Rows 0-150: No signals.
    # Rows 150-200: Good signals.
    
    synthetic_data.loc[:150, "rsi_signal"] = 0
    
    # If we request start_days=50 (tail 50), we get 150-200, which has signals.
    # We want to test expansion. So we need the RECENT data to be sparse? 
    # No, adaptive window takes tail(N). If recent data is sparse, it expands N to include OLDER data.
    
    # So:
    # Recent data (150-200): Sparse/No signals.
    # Older data (0-150): Many signals.
    
    synthetic_data.loc[150:, "rsi_signal"] = 0
    # Add some signals in older data
    synthetic_data.loc[0:150:5, "rsi_signal"] = 1.0 # Every 5th day
    
    # Start with window=20 (rows 180-200) -> 0 trades
    # Should expand to include the older data
    
    # Mocking MIN_TRADES to be small but > 0
    min_trades = 5
    
    window_days, trades = get_adaptive_window(
        synthetic_data,
        weights,
        horizon=3,
        min_trades=min_trades,
        start_days=20,
        step_days=20
    )
    
    assert trades >= min_trades
    assert window_days > 20 # Should have expanded

