"""
Tests for stock calculations module.
"""
import pytest
import pandas as pd
from datetime import datetime

from app.services.calculations import StockCalculations


class TestSafeFloat:
    """Tests for safe_float conversion."""
    
    def test_safe_float_with_number(self):
        calc = StockCalculations()
        assert calc.safe_float(42.5) == 42.5
    
    def test_safe_float_with_nan(self):
        calc = StockCalculations()
        assert calc.safe_float(float('nan')) == 0.0
    
    def test_safe_float_with_none(self):
        calc = StockCalculations()
        assert calc.safe_float(None) == 0.0
    
    def test_safe_float_with_series(self):
        calc = StockCalculations()
        series = pd.Series([1.5, 2.5, 3.5])
        assert calc.safe_float(series) == 1.5
    
    def test_safe_float_with_empty_series(self):
        calc = StockCalculations()
        series = pd.Series([], dtype=float)
        assert calc.safe_float(series) == 0.0
    
    def test_safe_float_with_string(self):
        calc = StockCalculations()
        assert calc.safe_float("invalid") == 0.0


class TestSafeInt:
    """Tests for safe_int conversion."""
    
    def test_safe_int_with_number(self):
        calc = StockCalculations()
        assert calc.safe_int(42) == 42
    
    def test_safe_int_with_float(self):
        calc = StockCalculations()
        assert calc.safe_int(42.7) == 42
    
    def test_safe_int_with_nan(self):
        calc = StockCalculations()
        assert calc.safe_int(float('nan')) == 0
    
    def test_safe_int_with_none(self):
        calc = StockCalculations()
        assert calc.safe_int(None) == 0


class TestCalculateSMA:
    """Tests for SMA calculation."""
    
    def test_sma_with_enough_data(self):
        calc = StockCalculations()
        prices = list(range(1, 201))  # 200 prices
        sma = calc.calculate_sma(prices, 200)
        assert sma == pytest.approx(100.5)  # Average of 1-200
    
    def test_sma_with_insufficient_data(self):
        calc = StockCalculations()
        prices = list(range(1, 30))  # Only 29 prices
        sma = calc.calculate_sma(prices, 200)
        assert sma is None
    
    def test_sma_with_fallback(self):
        calc = StockCalculations()
        prices = list(range(1, 51))  # 50 prices (fallback threshold)
        sma = calc.calculate_sma(prices, 200)
        assert sma == pytest.approx(25.5)  # Average of 1-50
    
    def test_sma_uses_last_n_prices(self):
        calc = StockCalculations()
        prices = [10] * 200 + [20] * 200  # 400 prices
        sma = calc.calculate_sma(prices, 200)
        assert sma == pytest.approx(20.0)  # Should use last 200


class TestPriceVsSMA:
    """Tests for price vs SMA percentage."""
    
    def test_price_above_sma(self):
        calc = StockCalculations()
        result = calc.calculate_price_vs_sma(110, 100)
        assert result == pytest.approx(10.0)
    
    def test_price_below_sma(self):
        calc = StockCalculations()
        result = calc.calculate_price_vs_sma(90, 100)
        assert result == pytest.approx(-10.0)
    
    def test_price_equals_sma(self):
        calc = StockCalculations()
        result = calc.calculate_price_vs_sma(100, 100)
        assert result == pytest.approx(0.0)
    
    def test_zero_sma(self):
        calc = StockCalculations()
        result = calc.calculate_price_vs_sma(100, 0)
        assert result == 0.0


class TestGainLoss:
    """Tests for gain/loss calculations."""
    
    def test_positive_gain(self):
        calc = StockCalculations()
        gain, pct = calc.calculate_gain_loss(
            investment_price=100,
            current_price=120,
            allocated_value=1000
        )
        assert gain == pytest.approx(200.0)  # $1000 -> $1200
        assert pct == pytest.approx(20.0)  # 20% gain
    
    def test_negative_loss(self):
        calc = StockCalculations()
        gain, pct = calc.calculate_gain_loss(
            investment_price=100,
            current_price=80,
            allocated_value=1000
        )
        assert gain == pytest.approx(-200.0)  # $1000 -> $800
        assert pct == pytest.approx(-20.0)  # 20% loss
    
    def test_no_change(self):
        calc = StockCalculations()
        gain, pct = calc.calculate_gain_loss(
            investment_price=100,
            current_price=100,
            allocated_value=1000
        )
        assert gain == pytest.approx(0.0)
        assert pct == pytest.approx(0.0)
    
    def test_missing_investment_price(self):
        calc = StockCalculations()
        gain, pct = calc.calculate_gain_loss(
            investment_price=None,
            current_price=100,
            allocated_value=1000
        )
        assert gain is None
        assert pct is None
    
    def test_zero_investment_price(self):
        calc = StockCalculations()
        gain, pct = calc.calculate_gain_loss(
            investment_price=0,
            current_price=100,
            allocated_value=1000
        )
        assert gain is None
        assert pct is None


class TestYTDReturn:
    """Tests for YTD return calculation."""
    
    def test_ytd_with_previous_year_data(self):
        calc = StockCalculations()
        current_year = datetime.now().year
        history = [
            {"date": f"{current_year - 1}-12-31", "close": 100},
            {"date": f"{current_year}-01-02", "close": 105},
            {"date": f"{current_year}-01-03", "close": 110},
        ]
        ytd = calc.calculate_ytd_return(110, history)
        assert ytd == pytest.approx(10.0)  # 100 -> 110 = 10%
    
    def test_ytd_with_only_current_year(self):
        calc = StockCalculations()
        current_year = datetime.now().year
        history = [
            {"date": f"{current_year}-01-02", "close": 100},
            {"date": f"{current_year}-01-03", "close": 110},
        ]
        ytd = calc.calculate_ytd_return(110, history)
        assert ytd == pytest.approx(10.0)  # Fallback to first day
    
    def test_ytd_empty_history(self):
        calc = StockCalculations()
        ytd = calc.calculate_ytd_return(110, [])
        assert ytd == 0.0
    
    def test_ytd_zero_current_price(self):
        calc = StockCalculations()
        ytd = calc.calculate_ytd_return(0, [{"date": "2024-01-01", "close": 100}])
        assert ytd == 0.0


class TestPeriodGain:
    """Tests for period gain calculation."""
    
    def test_positive_period_gain(self):
        calc = StockCalculations()
        history = [
            {"close": 102},
            {"close": 105},
            {"close": 110},
        ]
        gain = calc.calculate_period_gain(history, 100)
        assert gain == pytest.approx(10.0)
    
    def test_negative_period_gain(self):
        calc = StockCalculations()
        history = [
            {"close": 98},
            {"close": 95},
            {"close": 90},
        ]
        gain = calc.calculate_period_gain(history, 100)
        assert gain == pytest.approx(-10.0)
    
    def test_no_reference(self):
        calc = StockCalculations()
        history = [{"close": 110}]
        gain = calc.calculate_period_gain(history, None)
        assert gain is None
    
    def test_empty_history(self):
        calc = StockCalculations()
        gain = calc.calculate_period_gain([], 100)
        assert gain is None

