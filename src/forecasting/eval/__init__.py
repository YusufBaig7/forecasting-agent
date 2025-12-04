"""Evaluation modules."""

from .metrics import brier_score, expected_calibration_error
from .backtest import run_backtest, BacktestResult
from .trading import simulate_trading, TradingResult, kelly_fraction_binary, implied_decimal_odds_from_prob

__all__ = [
    "brier_score",
    "expected_calibration_error",
    "run_backtest",
    "BacktestResult",
    "simulate_trading",
    "TradingResult",
    "kelly_fraction_binary",
    "implied_decimal_odds_from_prob",
]
