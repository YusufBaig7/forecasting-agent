"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from forecasting.eval.metrics import brier_score, expected_calibration_error


class TestBrierScore:
    """Tests for Brier score calculation."""

    def test_perfect_predictions(self):
        """Perfect predictions should have Brier score of 0."""
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        score = brier_score(y_true, y_prob)
        assert score == 0.0

    def test_worst_predictions(self):
        """Worst predictions (always wrong) should have Brier score of 1."""
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        score = brier_score(y_true, y_prob)
        assert score == 1.0

    def test_uniform_predictions(self):
        """Uniform 0.5 predictions should have Brier score of 0.25."""
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        score = brier_score(y_true, y_prob)
        assert score == 0.25

    def test_shape_mismatch(self):
        """Should raise ValueError for shape mismatch."""
        y_true = np.array([1, 0])
        y_prob = np.array([0.5, 0.5, 0.5])
        with pytest.raises(ValueError, match="same shape"):
            brier_score(y_true, y_prob)

    def test_invalid_probabilities(self):
        """Should raise ValueError for probabilities outside [0, 1]."""
        y_true = np.array([1, 0])
        y_prob = np.array([1.5, 0.5])
        with pytest.raises(ValueError, match="between 0 and 1"):
            brier_score(y_true, y_prob)

        y_prob = np.array([-0.1, 0.5])
        with pytest.raises(ValueError, match="between 0 and 1"):
            brier_score(y_true, y_prob)


class TestExpectedCalibrationError:
    """Tests for Expected Calibration Error calculation."""

    def test_perfectly_calibrated(self):
        """Perfectly calibrated predictions should have ECE of 0."""
        # Create perfectly calibrated predictions
        n = 1000
        y_true = np.random.binomial(1, 0.7, n)
        y_prob = np.full(n, 0.7)
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        # Should be very close to 0 for large n
        assert ece < 0.05

    def test_empty_input(self):
        """Empty input should return 0."""
        y_true = np.array([])
        y_prob = np.array([])
        ece = expected_calibration_error(y_true, y_prob)
        assert ece == 0.0

    def test_shape_mismatch(self):
        """Should raise ValueError for shape mismatch."""
        y_true = np.array([1, 0])
        y_prob = np.array([0.5, 0.5, 0.5])
        with pytest.raises(ValueError, match="same shape"):
            expected_calibration_error(y_true, y_prob)

    def test_miscalibrated_predictions(self):
        """Miscalibrated predictions should have higher ECE."""
        # Predictions that are systematically too high
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
        ece = expected_calibration_error(y_true, y_prob, n_bins=5)
        # Should be high (poor calibration)
        assert ece > 0.5

    def test_different_bin_counts(self):
        """Should work with different numbers of bins."""
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.8, 0.2, 0.7, 0.3, 0.6, 0.4])

        ece_5 = expected_calibration_error(y_true, y_prob, n_bins=5)
        ece_10 = expected_calibration_error(y_true, y_prob, n_bins=10)
        ece_15 = expected_calibration_error(y_true, y_prob, n_bins=15)

        # All should be valid (non-negative)
        assert ece_5 >= 0.0
        assert ece_10 >= 0.0
        assert ece_15 >= 0.0

