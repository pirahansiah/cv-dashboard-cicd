"""Tests for the ML training pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from main import generate_data, train_model, save_model, load_model


@pytest.fixture()
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate sample data for testing."""
    return generate_data(n_samples=100, seed=42)


@pytest.fixture()
def trained_model(sample_data: tuple[np.ndarray, np.ndarray]) -> LinearRegression:
    """Train and return a model for testing."""
    X, y = sample_data
    model, _ = train_model(X, y)
    return model


class TestGenerateData:
    """Tests for the synthetic data generator."""

    def test_output_shapes(self) -> None:
        X, y = generate_data(n_samples=50)
        assert X.shape == (50, 1)
        assert y.shape == (50,)

    def test_linearity(self) -> None:
        X, y = generate_data(n_samples=10)
        # y = 2x + 1
        expected = 2 * X.ravel() + 1
        np.testing.assert_array_equal(y, expected)

    def test_dtypes(self) -> None:
        X, y = generate_data()
        assert X.dtype == np.float64
        assert y.dtype == np.float64


class TestTrainModel:
    """Tests for the model training function."""

    def test_model_type(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = sample_data
        model, _ = train_model(X, y)
        assert isinstance(model, LinearRegression)

    def test_perfect_fit(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = sample_data
        _, mse = train_model(X, y)
        assert mse < 1e-10, f"MSE too high for perfect linear data: {mse}"

    def test_coefficients(self, trained_model: LinearRegression) -> None:
        # For y = 2x + 1, coefficient should be ~2, intercept ~1
        assert abs(trained_model.coef_[0] - 2.0) < 0.01
        assert abs(trained_model.intercept_ - 1.0) < 0.01

    def test_custom_test_size(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = sample_data
        model, mse = train_model(X, y, test_size=0.3)
        assert isinstance(model, LinearRegression)
        assert mse < 1e-10

    def test_reproducibility(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = sample_data
        _, mse1 = train_model(X, y, random_state=42)
        _, mse2 = train_model(X, y, random_state=42)
        assert mse1 == mse2


class TestModelPersistence:
    """Tests for model save/load."""

    def test_save_load_roundtrip(
        self, trained_model: LinearRegression, tmp_path: object
    ) -> None:
        from pathlib import Path
        path = Path(str(tmp_path)) / "test_model.joblib"
        save_model(trained_model, path)
        loaded = load_model(path)
        assert isinstance(loaded, LinearRegression)
        np.testing.assert_array_equal(trained_model.coef_, loaded.coef_)

    def test_loaded_model_predicts(
        self, trained_model: LinearRegression, tmp_path: object
    ) -> None:
        from pathlib import Path
        path = Path(str(tmp_path)) / "test_model.joblib"
        save_model(trained_model, path)
        loaded = load_model(path)
        X_test = np.array([[50.0]])
        pred = loaded.predict(X_test)
        assert abs(pred[0] - 101.0) < 0.01  # y = 2*50 + 1


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline(self, tmp_path: object) -> None:
        from pathlib import Path
        path = Path(str(tmp_path)) / "e2e_model.joblib"
        X, y = generate_data(n_samples=200, seed=99)
        model, mse = train_model(X, y, test_size=0.2, random_state=99)
        save_model(model, path)
        loaded = load_model(path)
        X_test = np.array([[101.0], [102.0], [103.0]])
        preds = loaded.predict(X_test)
        expected = np.array([203.0, 205.0, 207.0])
        np.testing.assert_allclose(preds, expected, atol=0.01)
