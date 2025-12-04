"""Tests for trading simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.eval.trading import (
    implied_decimal_odds_from_prob,
    kelly_fraction_binary,
    simulate_trading,
    TradingResult,
)


class TestImpliedDecimalOdds:
    """Tests for implied_decimal_odds_from_prob."""

    def test_basic_cases(self):
        """Test basic probability to odds conversions."""
        assert implied_decimal_odds_from_prob(0.5) == 2.0
        assert implied_decimal_odds_from_prob(0.25) == 4.0
        assert implied_decimal_odds_from_prob(0.8) == 1.25
        assert abs(implied_decimal_odds_from_prob(0.33) - 3.0303) < 0.01

    def test_edge_cases(self):
        """Test edge cases."""
        # Very low probability
        assert implied_decimal_odds_from_prob(0.01) == 100.0
        
        # Very high probability
        assert abs(implied_decimal_odds_from_prob(0.99) - 1.0101) < 0.01

    def test_invalid_probabilities(self):
        """Test that invalid probabilities raise ValueError."""
        with pytest.raises(ValueError):
            implied_decimal_odds_from_prob(0.0)
        
        with pytest.raises(ValueError):
            implied_decimal_odds_from_prob(1.0)
        
        with pytest.raises(ValueError):
            implied_decimal_odds_from_prob(-0.1)
        
        with pytest.raises(ValueError):
            implied_decimal_odds_from_prob(1.1)


class TestKellyFraction:
    """Tests for kelly_fraction_binary."""

    def test_no_edge(self):
        """Test that no bet is placed when forecast equals market."""
        assert kelly_fraction_binary(0.5, 0.5) == 0.0
        assert kelly_fraction_binary(0.7, 0.7) == 0.0

    def test_negative_edge(self):
        """Test that no bet is placed when forecast is below market."""
        assert kelly_fraction_binary(0.3, 0.5) == 0.0
        assert kelly_fraction_binary(0.4, 0.6) == 0.0

    def test_positive_edge(self):
        """Test Kelly fraction for positive edge."""
        # Forecast 0.6, market 0.5: edge = 0.1, kelly = 0.1 / 0.5 = 0.2
        kelly = kelly_fraction_binary(0.6, 0.5)
        assert abs(kelly - 0.2) < 0.01
        
        # Forecast 0.8, market 0.5: edge = 0.3, kelly = 0.3 / 0.5 = 0.6
        kelly = kelly_fraction_binary(0.8, 0.5)
        assert abs(kelly - 0.6) < 0.01
        
        # Forecast 0.9, market 0.7: edge = 0.2, kelly = 0.2 / 0.3 ≈ 0.667
        kelly = kelly_fraction_binary(0.9, 0.7)
        assert abs(kelly - 0.667) < 0.01

    def test_max_fraction_clamp(self):
        """Test that max_fraction clamps the Kelly fraction."""
        # Large edge that would exceed max_fraction
        kelly = kelly_fraction_binary(0.95, 0.5, max_fraction=0.5)
        assert kelly == 0.5
        
        kelly = kelly_fraction_binary(0.95, 0.5, max_fraction=0.25)
        assert kelly == 0.25

    def test_edge_cases(self):
        """Test edge cases for Kelly fraction."""
        # Very small edge
        kelly = kelly_fraction_binary(0.501, 0.5)
        assert kelly > 0.0
        assert kelly < 0.01
        
        # Market very close to 1.0
        kelly = kelly_fraction_binary(0.99, 0.95)
        assert kelly > 0.0
        assert kelly <= 1.0

    def test_invalid_inputs(self):
        """Test that invalid inputs return 0."""
        assert kelly_fraction_binary(0.0, 0.5) == 0.0
        assert kelly_fraction_binary(1.0, 0.5) == 0.0
        assert kelly_fraction_binary(0.5, 0.0) == 0.0
        assert kelly_fraction_binary(0.5, 1.0) == 0.0


class TestSimulateTrading:
    """Tests for simulate_trading."""

    def test_perfect_forecasts(self):
        """Test with perfect forecasts (always correct)."""
        df = pd.DataFrame({
            "as_of": pd.date_range("2024-01-01", periods=5, freq="D"),
            "event_id": [f"event_{i}" for i in range(5)],
            "p_yes": [0.8, 0.7, 0.9, 0.6, 0.85],
            "market_prob": [0.5, 0.5, 0.5, 0.5, 0.5],  # Market underestimates
            "y": [1, 1, 1, 1, 1],  # All YES
        })
        
        result = simulate_trading(df)
        
        assert result.total_return > 0.0
        assert result.num_bets > 0
        assert len(result.bankroll_series) == len(df) + 1  # +1 for initial bankroll
        assert result.bankroll_series.iloc[-1] > 1.0  # Bankroll increased

    def test_all_wrong_forecasts(self):
        """Test with all wrong forecasts."""
        df = pd.DataFrame({
            "as_of": pd.date_range("2024-01-01", periods=3, freq="D"),
            "event_id": [f"event_{i}" for i in range(3)],
            "p_yes": [0.8, 0.7, 0.9],  # Forecast YES
            "market_prob": [0.5, 0.5, 0.5],
            "y": [0, 0, 0],  # All NO (we were wrong)
        })
        
        result = simulate_trading(df)
        
        assert result.total_return < 0.0
        assert result.bankroll_series.iloc[-1] < 1.0  # Bankroll decreased

    def test_edge_threshold(self):
        """Test that edge_threshold filters bets."""
        df = pd.DataFrame({
            "as_of": pd.date_range("2024-01-01", periods=3, freq="D"),
            "event_id": [f"event_{i}" for i in range(3)],
            "p_yes": [0.6, 0.55, 0.65],  # Small edges
            "market_prob": [0.5, 0.5, 0.5],
            "y": [1, 1, 1],
        })
        
        # No threshold: all bets placed
        result1 = simulate_trading(df, edge_threshold=0.0)
        assert result1.num_bets == 3
        
        # High threshold: no bets placed
        result2 = simulate_trading(df, edge_threshold=0.2)
        assert result2.num_bets == 0
        assert result2.total_return == 0.0

    def test_bankroll_update_correctness(self):
        """Test that bankroll updates are correct."""
        # Single bet: forecast 0.8, market 0.5, outcome YES
        # Kelly = (0.8 - 0.5) / (1 - 0.5) = 0.6
        # Decimal odds = 1/0.5 = 2.0, so b = 1.0
        # If YES: bankroll *= (1 + 0.6 * 1.0) = 1.6
        
        df = pd.DataFrame({
            "as_of": [pd.Timestamp("2024-01-01")],
            "event_id": ["event_0"],
            "p_yes": [0.8],
            "market_prob": [0.5],
            "y": [1],  # YES
        })
        
        result = simulate_trading(df)
        
        # Bankroll should be 1.6
        assert abs(result.bankroll_series.iloc[-1] - 1.6) < 0.01
        assert result.total_return == 0.6
        
        # Single bet: forecast 0.8, market 0.5, outcome NO
        # If NO: bankroll *= (1 - 0.6) = 0.4
        df2 = pd.DataFrame({
            "as_of": [pd.Timestamp("2024-01-01")],
            "event_id": ["event_0"],
            "p_yes": [0.8],
            "market_prob": [0.5],
            "y": [0],  # NO
        })
        
        result2 = simulate_trading(df2)
        assert abs(result2.bankroll_series.iloc[-1] - 0.4) < 0.01
        assert result2.total_return == -0.6

    def test_transaction_cost(self):
        """Test that transaction costs reduce returns."""
        df = pd.DataFrame({
            "as_of": pd.date_range("2024-01-01", periods=3, freq="D"),
            "event_id": [f"event_{i}" for i in range(3)],
            "p_yes": [0.8, 0.8, 0.8],
            "market_prob": [0.5, 0.5, 0.5],
            "y": [1, 1, 1],
        })
        
        result_no_cost = simulate_trading(df, transaction_cost=0.0)
        result_with_cost = simulate_trading(df, transaction_cost=0.05)  # 5% cost
        
        assert result_with_cost.total_return < result_no_cost.total_return

    def test_fractional_kelly(self):
        """Test that fractional Kelly reduces bet sizes."""
        df = pd.DataFrame({
            "as_of": [pd.Timestamp("2024-01-01")],
            "event_id": ["event_0"],
            "p_yes": [0.8],
            "market_prob": [0.5],
            "y": [1],
        })
        
        result_full = simulate_trading(df, kelly_fraction=1.0)
        result_half = simulate_trading(df, kelly_fraction=0.5)
        
        # Half Kelly should have smaller return (but still positive)
        assert result_half.total_return < result_full.total_return
        assert result_half.total_return > 0.0

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Create a series with known mean and std
        df = pd.DataFrame({
            "as_of": pd.date_range("2024-01-01", periods=10, freq="D"),
            "event_id": [f"event_{i}" for i in range(10)],
            "p_yes": [0.8] * 10,
            "market_prob": [0.5] * 10,
            "y": [1] * 10,  # All wins
        })
        
        result = simulate_trading(df)
        
        # All wins should give positive Sharpe
        assert result.sharpe > 0.0
        
        # Check that Sharpe is computed correctly
        returns = result.per_bet_returns
        if len(returns) > 0 and returns.std() > 0:
            expected_sharpe = (returns.mean() / returns.std()) * np.sqrt(len(returns))
            assert abs(result.sharpe - expected_sharpe) < 0.01

    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        # Create a sequence: win, lose, win, lose (should have drawdown)
        df = pd.DataFrame({
            "as_of": pd.date_range("2024-01-01", periods=4, freq="D"),
            "event_id": [f"event_{i}" for i in range(4)],
            "p_yes": [0.8, 0.8, 0.8, 0.8],
            "market_prob": [0.5, 0.5, 0.5, 0.5],
            "y": [1, 0, 1, 0],  # Win, lose, win, lose
        })
        
        result = simulate_trading(df)
        
        # Should have some drawdown
        assert result.max_drawdown >= 0.0
        assert result.max_drawdown <= 1.0
        
        # Verify drawdown is computed from peak
        bankroll = result.bankroll_series
        if len(bankroll) > 1:
            running_max = bankroll.expanding().max()
            drawdown = (bankroll - running_max) / running_max
            expected_max_dd = abs(drawdown.min())
            assert abs(result.max_drawdown - expected_max_dd) < 0.01

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame(columns=["as_of", "event_id", "p_yes", "market_prob", "y"])
        
        result = simulate_trading(df)
        
        assert result.total_return == 0.0
        assert result.sharpe == 0.0
        assert result.max_drawdown == 0.0
        assert result.num_bets == 0

    def test_missing_market_prob(self):
        """Test that rows without market_prob are skipped."""
        df = pd.DataFrame({
            "as_of": pd.date_range("2024-01-01", periods=3, freq="D"),
            "event_id": [f"event_{i}" for i in range(3)],
            "p_yes": [0.8, 0.8, 0.8],
            "market_prob": [0.5, None, 0.5],  # Middle one missing
            "y": [1, 1, 1],
        })
        
        result = simulate_trading(df)
        
        # Should only bet on 2 events (first and third)
        assert result.num_bets == 2

    def test_chronological_sorting(self):
        """Test that bets are processed in chronological order."""
        # Create events out of order
        df = pd.DataFrame({
            "as_of": [
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
            ],
            "event_id": ["event_2", "event_0", "event_1"],
            "p_yes": [0.8, 0.8, 0.8],
            "market_prob": [0.5, 0.5, 0.5],
            "y": [1, 1, 1],
        })
        
        result = simulate_trading(df)
        
        # Bankroll should increase monotonically (all wins)
        bankroll = result.bankroll_series
        assert all(bankroll.iloc[i] <= bankroll.iloc[i+1] for i in range(len(bankroll)-1))

