"""Trading simulator for binary markets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def implied_decimal_odds_from_prob(q: float) -> float:
    """
    Convert probability to implied decimal odds.
    
    Args:
        q: Probability in (0, 1)
        
    Returns:
        Decimal odds (>= 1.0)
        
    Raises:
        ValueError: If q not in (0, 1)
    """
    if q <= 0 or q >= 1:
        raise ValueError(f"Probability q must be in (0, 1), got {q}")
    return 1.0 / q


def kelly_fraction_binary(p: float, q: float, max_fraction: float = 1.0) -> float:
    """
    Compute Kelly fraction for binary bet using fair-odds approximation.
    
    For a bet on YES:
    - Forecast probability: p
    - Market probability: q
    - Decimal odds from market: b = (1/q) - 1
    - Kelly fraction: f* = (p*(b+1) - 1) / b = (p/q - 1) / ((1/q) - 1)
    
    Simplified: f* = (p - q) / (1 - q)
    
    Args:
        p: Forecast probability (0 < p < 1)
        q: Market probability (0 < q < 1)
        max_fraction: Maximum fraction to bet (default: 1.0 = full Kelly)
        
    Returns:
        Kelly fraction clamped to [0, max_fraction]
    """
    if p <= 0 or p >= 1 or q <= 0 or q >= 1:
        return 0.0
    
    # Edge: p - q (positive if we think market underestimates)
    edge = p - q
    
    # Only bet if we have positive edge
    if edge <= 0:
        return 0.0
    
    # Kelly fraction: (p - q) / (1 - q)
    # This is the simplified form for binary bets
    kelly = edge / (1.0 - q)
    
    # Clamp to [0, max_fraction]
    return max(0.0, min(kelly, max_fraction))


@dataclass
class TradingResult:
    """Results from trading simulation."""
    total_return: float
    sharpe: float
    max_drawdown: float
    num_bets: int
    bankroll_series: pd.Series
    per_bet_returns: pd.Series


def simulate_trading(
    df: pd.DataFrame,
    forecast_col: str = "p_yes",
    market_col: str = "market_prob",
    outcome_col: str = "y",
    edge_threshold: float = 0.0,
    kelly_fraction: float = 1.0,
    max_fraction: float = 1.0,
    transaction_cost: float = 0.0,
) -> TradingResult:
    """
    Simulate trading strategy on binary market forecasts.
    
    Strategy:
    - Only bet if edge (forecast - market) exceeds threshold
    - Use fractional Kelly sizing
    - Sort chronologically (as_of, then event_id)
    - Start with bankroll = 1.0
    
    Args:
        df: DataFrame with columns: as_of, event_id, forecast_col, market_col, outcome_col
        forecast_col: Column name for forecast probability
        market_col: Column name for market probability
        outcome_col: Column name for outcome (1=YES, 0=NO)
        edge_threshold: Minimum edge required to place bet (default: 0.0)
        kelly_fraction: Fraction of Kelly to use (default: 1.0 = full Kelly)
        max_fraction: Maximum fraction of bankroll to bet (default: 1.0)
        transaction_cost: Transaction cost as fraction of bet size (default: 0.0)
        
    Returns:
        TradingResult with metrics and series
    """
    # Sort chronologically: as_of, then event_id
    df_sorted = df.sort_values(["as_of", "event_id"]).copy()
    
    # Filter rows with valid market probabilities and outcomes
    df_valid = df_sorted.dropna(subset=[forecast_col, market_col, outcome_col]).copy()
    
    if len(df_valid) == 0:
        return TradingResult(
            total_return=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            num_bets=0,
            bankroll_series=pd.Series(dtype=float),
            per_bet_returns=pd.Series(dtype=float),
        )
    
    bankroll = 1.0
    bankroll_history = [bankroll]
    per_bet_returns_list = []
    
    for _, row in df_valid.iterrows():
        p = float(row[forecast_col])
        q = float(row[market_col])
        y = int(row[outcome_col])
        
        # Compute edge
        edge = p - q
        
        # Only bet if edge exceeds threshold
        if edge <= edge_threshold:
            bankroll_history.append(bankroll)
            per_bet_returns_list.append(0.0)  # No bet, no return
            continue
        
        # Compute Kelly fraction
        full_kelly = kelly_fraction_binary(p, q, max_fraction=1.0)
        bet_fraction = full_kelly * kelly_fraction
        bet_fraction = min(bet_fraction, max_fraction)
        
        if bet_fraction <= 0:
            bankroll_history.append(bankroll)
            per_bet_returns_list.append(0.0)
            continue
        
        # Apply transaction cost
        effective_fraction = bet_fraction * (1.0 - transaction_cost)
        
        # Compute decimal odds from market probability
        decimal_odds = implied_decimal_odds_from_prob(q)
        b = decimal_odds - 1.0  # Net odds (what you win per unit bet)
        
        # Place bet and update bankroll
        if y == 1:  # YES outcome
            # Win: bankroll increases by f * b
            bankroll *= (1.0 + effective_fraction * b)
        else:  # NO outcome (y == 0)
            # Lose: bankroll decreases by f
            bankroll *= (1.0 - effective_fraction)
        
        # Compute per-bet return
        if y == 1:
            bet_return = effective_fraction * b
        else:
            bet_return = -effective_fraction
        
        bankroll_history.append(bankroll)
        per_bet_returns_list.append(bet_return)
    
    # Compute metrics
    total_return = bankroll - 1.0
    
    # Sharpe ratio: mean(returns) / std(returns) * sqrt(N)
    per_bet_returns_series = pd.Series(per_bet_returns_list)
    if len(per_bet_returns_series) > 0:
        mean_return = per_bet_returns_series.mean()
        std_return = per_bet_returns_series.std()
        if std_return > 0:
            sharpe = (mean_return / std_return) * np.sqrt(len(per_bet_returns_series))
        else:
            sharpe = 0.0 if mean_return == 0.0 else float('inf')
    else:
        sharpe = 0.0
    
    # Max drawdown: maximum peak-to-trough decline
    bankroll_series = pd.Series(bankroll_history)
    if len(bankroll_series) > 0:
        running_max = bankroll_series.expanding().max()
        drawdown = (bankroll_series - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
    else:
        max_drawdown = 0.0
    
    return TradingResult(
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        num_bets=sum(1 for r in per_bet_returns_list if r != 0.0),
        bankroll_series=bankroll_series,
        per_bet_returns=per_bet_returns_series,
    )

