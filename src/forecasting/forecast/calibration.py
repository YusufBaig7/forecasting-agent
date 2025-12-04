"""Calibration for probability forecasts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from forecasting.forecast.baseline_market import MarketBaselineForecaster
from forecasting.models import Event, Forecast, MarketSnapshot


def logit(p: float) -> float:
    """
    Compute logit transform: log(p / (1 - p)).
    
    Clamps p to [1e-6, 1-1e-6] to avoid numerical issues.
    
    Args:
        p: Probability in [0, 1]
        
    Returns:
        Logit value
    """
    p_clamped = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p_clamped / (1.0 - p_clamped))


def sigmoid(z: float) -> float:
    """
    Compute sigmoid transform: 1 / (1 + exp(-z)).
    
    Args:
        z: Logit value
        
    Returns:
        Probability in (0, 1)
    """
    # Clamp z to avoid overflow
    z_clamped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clamped))


@dataclass
class PlattCalibrator:
    """
    Platt scaling calibrator using logistic regression.
    
    Fits: sigmoid(a * logit(p) + b) where a, b are learned parameters.
    """

    a: float = 1.0
    b: float = 0.0
    fitted: bool = False

    def fit(self, p_pred: np.ndarray, y: np.ndarray) -> None:
        """
        Fit calibrator on predictions and true outcomes.
        
        Args:
            p_pred: Predicted probabilities (shape: [n_samples])
            y: True binary outcomes (shape: [n_samples], values in {0, 1})
        """
        p_pred = np.asarray(p_pred, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if len(p_pred) != len(y):
            raise ValueError("p_pred and y must have the same length")
        
        if len(p_pred) == 0:
            raise ValueError("Cannot fit on empty data")
        
        # Clamp predictions to avoid edge cases
        p_pred_clamped = np.clip(p_pred, 1e-6, 1.0 - 1e-6)
        
        # Transform to logit space
        logit_p = np.array([logit(p) for p in p_pred_clamped]).reshape(-1, 1)
        
        # Fit logistic regression with strong regularization
        # C=1e-3 gives strong regularization (smaller C = stronger regularization)
        model = LogisticRegression(C=1e-3, max_iter=1000, solver="lbfgs")
        model.fit(logit_p, y)
        
        # Extract coefficients: model predicts sigmoid(coef * x + intercept)
        # We want: sigmoid(a * logit(p) + b)
        self.a = float(model.coef_[0][0])
        self.b = float(model.intercept_[0])
        self.fitted = True

    def predict(self, p_pred: np.ndarray) -> np.ndarray:
        """
        Calibrate predictions.
        
        Args:
            p_pred: Predicted probabilities (shape: [n_samples])
            
        Returns:
            Calibrated probabilities (shape: [n_samples])
        """
        if not self.fitted:
            raise RuntimeError("Calibrator must be fitted before prediction")
        
        p_pred = np.asarray(p_pred, dtype=float)
        p_pred_clamped = np.clip(p_pred, 1e-6, 1.0 - 1e-6)
        
        # Apply calibration: sigmoid(a * logit(p) + b)
        calibrated = np.array([
            sigmoid(self.a * logit(p) + self.b) for p in p_pred_clamped
        ])
        
        return calibrated

    def save(self, path: Path) -> None:
        """Save calibrator parameters to JSON file."""
        data = {
            "a": self.a,
            "b": self.b,
            "fitted": self.fitted,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> PlattCalibrator:
        """Load calibrator parameters from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(a=data["a"], b=data["b"], fitted=data["fitted"])


@dataclass
class Extremizer:
    """
    Extremizer that pushes probabilities away from 0.5.
    
    Applies: sigmoid(alpha * logit(p))
    - alpha > 1: makes probabilities more extreme (closer to 0 or 1)
    - alpha < 1: makes probabilities less extreme (closer to 0.5)
    """

    alpha: float = 1.0

    def __post_init__(self):
        """Validate alpha parameter."""
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")

    def predict(self, p_pred: np.ndarray) -> np.ndarray:
        """
        Extremize predictions.
        
        Args:
            p_pred: Predicted probabilities (shape: [n_samples])
            
        Returns:
            Extremized probabilities (shape: [n_samples])
        """
        p_pred = np.asarray(p_pred, dtype=float)
        p_pred_clamped = np.clip(p_pred, 1e-6, 1.0 - 1e-6)
        
        # Apply extremization: sigmoid(alpha * logit(p))
        extremized = np.array([
            sigmoid(self.alpha * logit(p)) for p in p_pred_clamped
        ])
        
        return extremized


class CalibratedForecaster:
    """
    Wrapper that applies calibration and optional extremization to any forecaster.
    """

    def __init__(
        self,
        base_forecaster,  # Any forecaster with predict() and model_name
        calibrator: Optional[PlattCalibrator] = None,
        extremizer: Optional[Extremizer] = None,
    ):
        """
        Initialize calibrated forecaster.
        
        Args:
            base_forecaster: Base forecaster to wrap
            calibrator: Optional calibrator to apply
            extremizer: Optional extremizer to apply (applied after calibration)
        """
        self.base_forecaster = base_forecaster
        self.calibrator = calibrator
        self.extremizer = extremizer

    @property
    def model_name(self) -> str:
        """Get model name with calibration suffix."""
        name = self.base_forecaster.model_name
        if self.calibrator is not None:
            name += "+cal"
        if self.extremizer is not None:
            name += f"+ext{self.extremizer.alpha:.2f}"
        return name

    def predict(
        self, event: Event, snapshot: MarketSnapshot, as_of: datetime
    ) -> Forecast:
        """
        Generate calibrated forecast.
        
        Args:
            event: Event to forecast
            snapshot: Market snapshot
            as_of: Timestamp for forecast
            
        Returns:
            Calibrated Forecast
        """
        # Get base forecast
        base_forecast = self.base_forecaster.predict(event, snapshot, as_of)
        
        # Apply calibration if available
        if self.calibrator is not None:
            p_cal = self.calibrator.predict(np.array([base_forecast.p_yes]))[0]
            base_forecast = Forecast(
                event_id=base_forecast.event_id,
                as_of=base_forecast.as_of,
                p_yes=float(p_cal),
                model=self.model_name,
                rationale=base_forecast.rationale,
                metadata={
                    **base_forecast.metadata,
                    "base_p_yes": base_forecast.p_yes,
                    "calibrated": True,
                },
            )
        
        # Apply extremization if available (after calibration)
        if self.extremizer is not None:
            p_ext = self.extremizer.predict(np.array([base_forecast.p_yes]))[0]
            base_forecast = Forecast(
                event_id=base_forecast.event_id,
                as_of=base_forecast.as_of,
                p_yes=float(p_ext),
                model=self.model_name,
                rationale=base_forecast.rationale,
                metadata={
                    **base_forecast.metadata,
                    "pre_extremized_p_yes": base_forecast.p_yes,
                    "extremized": True,
                },
            )
        
        return base_forecast

