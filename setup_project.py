#!/usr/bin/env python3
"""Generate a small, runnable Cats vs. Dogs SVM project."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path


IMAGE_WIDTH = 16
IMAGE_HEIGHT = 16
CHANNELS = 3
SAMPLES_PER_CLASS = 100
SEED = 42

REQUIREMENTS = """\
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
"""

README = """\
# Cats vs. Dogs Image Classification with SVM

## Project goal

This internship Task-3 project demonstrates image classification with a
scikit-learn Support Vector Machine (SVM). It includes a deterministic,
synthetic dataset of 100 cat-like and 100 dog-like RGB pixel images so the
workflow can be tested immediately without downloading a large image dataset.
The mock images are flattened into pixel columns in `dataset/mock_images.csv`.
They are for pipeline verification, not for measuring real-world accuracy.

## Classification architecture

1. Load flattened RGB pixels and integer labels (`0 = Cat`, `1 = Dog`).
2. Coerce pixel columns to numeric values, replace infinities, and impute
   missing pixel values with each column's median (or zero if entirely empty).
3. Split the data into stratified 80% training and 20% validation subsets.
4. Standardize features and train an RBF-kernel SVC in a scikit-learn pipeline.
5. Print the classification report, confusion matrix, accuracy, and
   macro/weighted precision and recall.

## Setup and run

From this project directory, run:

```console
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\\.venv\\Scripts\\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies and train:

```console
python -m pip install -r requirements.txt
python src/train_svm.py
```

To use another CSV with the same `label` plus flattened pixel-column format:

```console
python src/train_svm.py --data path/to/images.csv
```

## Dataset format

`dataset/mock_images.csv` contains one image per row. The first column,
`label`, is `0` for Cats or `1` for Dogs; the remaining columns contain the
flattened 16 x 16 x 3 RGB pixel values (0-255). The generated data is
deterministic and balanced.
"""

TRAIN_SCRIPT = '''\
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


def run(data_path: Path) -> None:
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
        SVC(kernel="rbf", C=10.0, gamma="scale", random_state=RANDOM_STATE),
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)

    print(f"Loaded {len(y)} samples with {X.shape[1]} flattened pixel features.")
    print(f"Training samples: {len(y_train)} | Validation samples: {len(y_valid)}")
    print("\\nClassification report:")
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
    args = parser.parse_args()
    try:
        run(args.data)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def make_mock_image(label: int, rng: random.Random) -> list[int]:
    """Return one flattened RGB image with simple class-specific structure."""
    pixels: list[int] = []
    for y in range(IMAGE_HEIGHT):
        for x in range(IMAGE_WIDTH):
            dx, dy = x - 7.5, y - 7.5
            face = dx * dx + dy * dy < 34
            if label == 0:
                left_ear = y < 7 and abs(x - (3 + y // 2)) <= 1
                right_ear = y < 7 and abs(x - (12 - y // 2)) <= 1
                ear = left_ear or right_ear
            else:
                left_ear = x < 5 and 4 <= y <= 10
                right_ear = x > 10 and 4 <= y <= 10
                ear = left_ear or right_ear

            eye = y in (7, 8) and x in (5, 10)
            nose = y == 10 and 7 <= x <= 8
            if eye or nose:
                color = (25, 25, 25)
            elif ear:
                color = (155, 95, 90) if label == 0 else (105, 80, 65)
            elif face:
                color = (205, 155, 145) if label == 0 else (185, 145, 105)
            else:
                color = (75, 115, 165) if label == 0 else (90, 145, 95)

            pixels.extend(
                max(0, min(255, channel + rng.randint(-12, 12)))
                for channel in color
            )
    return pixels


def write_dataset(path: Path) -> None:
    rng = random.Random(SEED)
    feature_count = IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS
    header = ["label"] + [f"pixel_{index:04d}" for index in range(feature_count)]
    with path.open("w", newline="", encoding="utf-8") as dataset_file:
        writer = csv.writer(dataset_file)
        writer.writerow(header)
        for label in (0, 1):
            for _ in range(SAMPLES_PER_CLASS):
                writer.writerow([label, *make_mock_image(label, rng)])


def build_project(project_dir: Path, force: bool) -> None:
    generated_files = {
        "dataset/mock_images.csv": None,
        "src/train_svm.py": TRAIN_SCRIPT,
        "requirements.txt": REQUIREMENTS,
        "README.md": README,
    }
    conflicts = [
        project_dir / relative_path
        for relative_path in generated_files
        if (project_dir / relative_path).exists()
    ]
    if conflicts and not force:
        paths = "\n".join(f"  - {path}" for path in conflicts)
        raise FileExistsError(
            "Refusing to overwrite existing project files. Choose another "
            "output directory or pass --force:\n" + paths
        )

    (project_dir / "dataset").mkdir(parents=True, exist_ok=True)
    (project_dir / "src").mkdir(parents=True, exist_ok=True)
    for relative_path, content in generated_files.items():
        destination = project_dir / relative_path
        if relative_path == "dataset/mock_images.csv":
            write_dataset(destination)
        else:
            destination.write_text(content or "", encoding="utf-8", newline="\n")


def print_commands(project_dir: Path) -> None:
    quoted_dir = f'"{project_dir}"'
    generator_path = Path(__file__).resolve()
    print(
        f"""
Project created at: {project_dir}

To run the generator again:
  python "{generator_path}" --output-dir {quoted_dir}

Run these commands in Windows PowerShell:
  cd {quoted_dir}
  python -m venv .venv
  .\\.venv\\Scripts\\Activate.ps1
  python -m pip install -r requirements.txt
  python src/train_svm.py
  git init -b main
  git add .
  git commit -m "Initial commit: Cats vs Dogs SVM classifier"
  gh repo create PRODIGY_ML_03 --public --source=. --remote=origin --push

On macOS/Linux, activate the environment with:
  source .venv/bin/activate
"""
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a mock-data Cats vs. Dogs SVM project."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "PRODIGY_ML_03",
        help="destination directory (default: ./PRODIGY_ML_03)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the generated files if they already exist",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_dir = args.output_dir.expanduser().resolve()
    try:
        build_project(project_dir, args.force)
    except (FileExistsError, OSError) as error:
        print(f"Setup failed: {error}", file=sys.stderr)
        return 1
    print_commands(project_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
