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
.\.venv\Scripts\Activate.ps1
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
