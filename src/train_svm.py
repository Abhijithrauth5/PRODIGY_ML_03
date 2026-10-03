"""Train and evaluate an SVM on flattened Cats vs. Dogs pixel data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLASS_NAMES = ["Cats", "Dogs"]
RANDOM_STATE = 42


def load_and_clean_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load flattened pixel columns and clean invalid feature values."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    if "label" not in frame.columns:
        raise ValueError("Dataset must contain a 'label' column.")
    if len(frame.columns) < 2:
        raise ValueError("Dataset must contain at least one pixel feature column.")

    labels = pd.to_numeric(frame["label"], errors="coerce")
    valid_labels = labels.isin([0, 1])
    removed = int((~valid_labels).sum())
    frame = frame.loc[valid_labels].copy()
    y = labels.loc[valid_labels].astype(int).to_numpy()
    if removed:
        print(f"Removed {removed} row(s) with missing or unsupported labels.")

    if y.size == 0 or set(np.unique(y)) != {0, 1}:
        raise ValueError("Dataset must contain at least one valid example of each class (0 and 1).")

    pixels = frame.drop(columns="label").apply(pd.to_numeric, errors="coerce")
    pixels = pixels.replace([np.inf, -np.inf], np.nan)
    pixels = pixels.fillna(pixels.median()).fillna(0)
    X = pixels.to_numpy(dtype=np.float32).reshape(len(pixels), -1)

    if not np.isfinite(X).all():
        raise ValueError("Pixel features still contain non-finite values after cleaning.")
    class_counts = np.bincount(y, minlength=2)
    if np.any(class_counts < 2):
        raise ValueError("Each class needs at least two examples for a stratified split.")
    return X, y


def run(data_path: Path, kernel: str = "rbf") -> None:
    X, y = load_and_clean_data(data_path)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel, C=10.0, gamma="scale", random_state=RANDOM_STATE),
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)

    print(f"Loaded {len(y)} samples with {X.shape[1]} flattened pixel features.")
    print(f"Training samples: {len(y_train)} | Validation samples: {len(y_valid)}")
    print(f"SVC kernel: {kernel}")
    print("\nClassification report:")
    print(
        classification_report(
            y_valid,
            predictions,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=true, columns=predicted; order: Cats, Dogs):")
    print(confusion_matrix(y_valid, predictions, labels=[0, 1]))
    print(f"Accuracy: {accuracy_score(y_valid, predictions):.4f}")
    print(
        "Precision (macro / weighted): "
        f"{precision_score(y_valid, predictions, average='macro', zero_division=0):.4f} / "
        f"{precision_score(y_valid, predictions, average='weighted', zero_division=0):.4f}"
    )
    print(
        "Recall (macro / weighted): "
        f"{recall_score(y_valid, predictions, average='macro', zero_division=0):.4f} / "
        f"{recall_score(y_valid, predictions, average='weighted', zero_division=0):.4f}"
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "dataset" / "mock_images.csv",
        help="CSV with a label column followed by flattened pixel features.",
    )
    parser.add_argument(
        "--kernel",
        choices=("linear", "rbf"),
        default="rbf",
        help="SVC kernel to use (default: rbf).",
    )
    args = parser.parse_args()
    try:
        run(args.data, kernel=args.kernel)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
