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
# Cat vs. Dog Image Classification using Support Vector Machines (SVM)

## Project overview

This project demonstrates a complete image-classification workflow that
distinguishes cats from dogs using scikit-learn's Support Vector Classifier
(`SVC`). It includes a small, deterministic mock pixel dataset for immediate
pipeline verification without a large download. The synthetic data is for
testing the workflow only; its scores do not represent real-world accuracy.

## Objective

Build an image classification system utilizing a linear or RBF-kernel Support
Vector Machine. The workflow cleans and flattens pixel features, creates a
stratified 80/20 training/validation split, scales the features, trains an
SVC, and reports per-class precision/recall, a classification report, and a
confusion matrix. The RBF kernel is used by default; select `--kernel linear`
to run a linear SVM.

## Project architecture

```text
PRODIGY_ML_03/
├── dataset/
│   └── mock_images.csv       # 100 mock Cats (0) + 100 mock Dogs (1)
├── src/
│   └── train_svm.py          # Data cleaning, SVC training, validation metrics
├── requirements.txt          # Python dependencies
├── setup_project.py          # Regenerates the project and mock dataset
└── README.md
```

Each CSV row represents one flattened 16 x 16 RGB image. The `label` column
uses `0` for Cats and `1` for Dogs; remaining columns contain pixel values.
Missing or invalid pixel values are cleaned before the stratified split.
Standardization and SVC training are combined in a scikit-learn pipeline to
avoid fitting preprocessing on validation samples.

## Installation

Use Python 3.9 or newer. From the repository root, install the dependencies:

```bash
pip install -r requirements.txt
```

Optionally create and activate a virtual environment before installing:

```bash
python -m venv .venv
# Windows PowerShell
.\\.venv\\Scripts\\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

## Run

Run the default RBF SVM and print its validation report and confusion matrix:

```bash
python src/train_svm.py
```

Choose the linear kernel or provide a different CSV in the same format:

```bash
python src/train_svm.py --kernel linear
python src/train_svm.py --data path/to/images.csv
```

The evaluation output includes accuracy, the classification report, a
confusion matrix (true labels by rows and predictions by columns), and
macro/weighted precision and recall.
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
