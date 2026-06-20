"""ML training pipeline — linear regression on synthetic data.

Generates synthetic y = 2x + 1 data, trains a LinearRegression model,
evaluates MSE, and saves the trained model to disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def generate_data(n_samples: int = 100, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic linear data y = 2x + 1.

    Args:
        n_samples: Number of data points.
        seed: Random seed for train/test split.

    Returns:
        Tuple of (X, y) arrays.
    """
    x = np.array([[i] for i in range(1, n_samples + 1)], dtype=np.float64)
    y = np.array([2 * i + 1 for i in range(1, n_samples + 1)], dtype=np.float64)
    return x, y


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[LinearRegression, float]:
    """Train a LinearRegression model and return it with its MSE.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Fraction of data for testing.
        random_state: Random state for reproducibility.

    Returns:
        Tuple of (trained model, test MSE).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    return model, mse


def save_model(model: LinearRegression, path: Path) -> None:
    """Save the trained model to disk.

    Args:
        model: Trained scikit-learn model.
        path: Output path for the joblib file.
    """
    joblib.dump(model, path)


def load_model(path: Path) -> LinearRegression:
    """Load a trained model from disk.

    Args:
        path: Path to the joblib file.

    Returns:
        Loaded model.
    """
    return joblib.load(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a linear regression model")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("model.joblib"),
        help="Path to save the trained model",
    )
    parser.add_argument(
        "-n", "--n-samples",
        type=int,
        default=100,
        help="Number of training samples",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point — train, evaluate, and save the model."""
    args = parse_args(argv)

    X, y = generate_data(args.n_samples)
    model, mse = train_model(X, y, test_size=args.test_size, random_state=args.seed)

    print(f"Mean Squared Error: {mse:.6f}")
    save_model(model, args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
