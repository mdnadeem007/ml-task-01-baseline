# Task 01: Data Preparation and Baseline

A first, deliberately simple machine learning model on a public dataset, with a trivial baseline to beat.

## Dataset

**Breast Cancer Wisconsin (Diagnostic)** from the UCI Machine Learning Repository. It is bundled with scikit-learn (`sklearn.datasets.load_breast_cancer`), so there is nothing to download.

- 569 rows, 30 numeric features
- Binary target: malignant (0) or benign (1)
- No missing values, no duplicate rows
- Mildly imbalanced: about 63% benign

## What the notebook does

1. Loads the data and checks for missing values, duplicates and class balance
2. **Splits into train and test first** (80/20, stratified, `random_state=42`)
3. Builds a **trivial baseline**: always predict the most common class
4. Trains **one simple model**: `StandardScaler` + `LogisticRegression` in a single pipeline, so the scaler only ever sees training data
5. Reports **one accuracy number** on the held-out test set

## Results

| Model | Test accuracy |
|---|---|
| Baseline (most frequent class) | 0.632 |
| Logistic regression (scaled) | 0.982 |

The trusted number is **98.2% test accuracy** (112 of 114 correct), against a baseline of 63.2%.

## Decisions

- **Split before anything else.** The test set is used once, for the final scores.
- **Stratified split** because the classes are imbalanced.
- **Baseline is "most frequent class"**, not a coin flip, since a model that always says "benign" already scores about 63%.
- **Pipeline for scaling** to avoid leaking test-set statistics into training.
- **Limitation:** the test set has only 114 rows, so one more mistake changes accuracy by about 0.9 points. Cross-validation would be a good next step.
- **Out of scope on purpose:** tuning, model comparison and feature engineering.

## How to run

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
jupyter notebook baseline.ipynb
```

Then choose **Kernel > Restart & Run All**. It runs in a few seconds and needs no internet connection.

Alternatively, upload `baseline.ipynb` to Google Colab and run all cells.

## Files

- `baseline.ipynb`: the notebook
- `requirements.txt`: dependencies
- `README.md`: this file
