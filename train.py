"""
train.py
--------
Train a lightweight downstream classifier on pre-extracted Wav2Vec 2.0
embeddings and persist the best model to disk.

Author : Ayush Patel, IIIT Guwahati
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------
LABEL_MAP = {"bonafide": 1, "spoof": 0}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def build_model_and_grid(model_type: str):
    """
    Returns (estimator, param_grid) for the requested model type.
    All param_grids follow the paper's sweep ranges.
    """
    if model_type == "SVM":
        return (
            SVC(probability=True, kernel="rbf", random_state=42),
            {"C": [0.1, 0.2, 1.0]},
        )
    if model_type == "LogisticRegression":
        return (
            LogisticRegression(max_iter=2000, random_state=42),
            {"C": [0.1, 0.2, 10.0]},
        )
    if model_type == "KNN":
        return (
            KNeighborsClassifier(algorithm="auto"),
            {"n_neighbors": [3, 5, 6]},
        )
    if model_type == "NaiveBayes":
        return (GaussianNB(), {"var_smoothing": [1e-9]})
    if model_type == "DecisionTree":
        return (
            DecisionTreeClassifier(random_state=42),
            {
                "criterion": ["gini", "entropy"],
                "max_depth": [50, 100, 150],
            },
        )
    if model_type == "MLP":
        return (
            MLPClassifier(max_iter=500, random_state=42),
            {
                "hidden_layer_sizes": [(50,), (100,)],
                "activation": ["relu"],
                "batch_size": [32, 64],
                "learning_rate": ["constant", "invscaling"],
                "alpha": [0.0001],
            },
        )
    if model_type == "XGBoost":
        return (
            xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            ),
            {"max_depth": [5, 10, 15], "learning_rate": [0.1, 0.01]},
        )
    if model_type == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [50, 75, 100]},
        )
    raise ValueError(f"Unknown model type: '{model_type}'. "
                     "Choose from: SVM, LogisticRegression, KNN, NaiveBayes, "
                     "DecisionTree, MLP, XGBoost, RandomForest")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a pre-extracted embedding CSV.

    Expects columns:  [file_id?, feature_0 … feature_N-1, label]

    Returns (X: float32 array, y: int array).
    """
    df = pd.read_csv(csv_path)

    # Drop non-feature columns
    drop_cols = [c for c in ["file_id", "file_name"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Encode labels
    df["label"] = df["label"].map(LABEL_MAP)
    if df["label"].isna().any():
        unknown = df.loc[df["label"].isna(), "label"].unique().tolist()
        raise ValueError(f"Unknown label values found: {unknown}. "
                         "Expected 'bonafide' or 'spoof'.")

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df["label"].values.astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    train_csv: Path,
    dev_csv: Path,
    model_type: str,
    output_dir: Path,
    scale: bool = True,
    n_jobs: int = -1,
) -> dict:
    """
    Full training loop with grid-search over a predefined validation split.

    Returns a metrics dict.
    """
    logger.info("Loading training split  : %s", train_csv)
    X_train, y_train = load_split(train_csv)

    logger.info("Loading development split: %s", dev_csv)
    X_dev, y_dev = load_split(dev_csv)

    logger.info(
        "Shapes  — train: %s | dev: %s | classes: %s",
        X_train.shape, X_dev.shape, np.unique(y_train),
    )

    # ---- Optional StandardScaler ----------------------------------------
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_dev   = scaler.transform(X_dev)
        logger.info("Applied StandardScaler.")

    # ---- Build estimator + grid ------------------------------------------
    estimator, param_grid = build_model_and_grid(model_type)
    logger.info("Model type  : %s", model_type)
    logger.info("Param grid  : %s", param_grid)

    # ---- PredefinedSplit: -1 = train, 0 = validation --------------------
    X_combined = np.vstack([X_train, X_dev])
    y_combined = np.hstack([y_train, y_dev])

    test_fold = np.concatenate([
        np.full(len(X_train), -1, dtype=int),
        np.zeros(len(X_dev),       dtype=int),
    ])
    ps = PredefinedSplit(test_fold=test_fold)

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1",
        cv=ps,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
    )

    t0 = time.perf_counter()
    gs.fit(X_combined, y_combined)
    elapsed = time.perf_counter() - t0

    best = gs.best_estimator_
    logger.info("Best params  : %s  (grid-search time: %.1fs)", gs.best_params_, elapsed)

    # ---- Validation metrics ---------------------------------------------
    y_pred_dev = best.predict(X_dev)
    val_metrics = {
        "accuracy": round(float(accuracy_score(y_dev, y_pred_dev)), 6),
        "precision": round(float(precision_score(y_dev, y_pred_dev, zero_division=0)), 6),
        "recall": round(float(recall_score(y_dev, y_pred_dev, zero_division=0)), 6),
        "f1": round(float(f1_score(y_dev, y_pred_dev, zero_division=0)), 6),
    }
    logger.info(
        "Val  — acc: %.4f  prec: %.4f  rec: %.4f  F1: %.4f",
        val_metrics["accuracy"], val_metrics["precision"],
        val_metrics["recall"],   val_metrics["f1"],
    )

    # ---- Training metrics -----------------------------------------------
    y_pred_train = best.predict(X_train)
    train_metrics = {
        "accuracy": round(float(accuracy_score(y_train, y_pred_train)), 6),
        "f1":       round(float(f1_score(y_train, y_pred_train, zero_division=0)), 6),
    }
    logger.info(
        "Train — acc: %.4f  F1: %.4f",
        train_metrics["accuracy"], train_metrics["f1"],
    )

    # ---- Persist model + scaler -----------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path  = output_dir / f"{model_type}_model.pkl"
    scaler_path = output_dir / f"{model_type}_scaler.pkl" if scale else None

    joblib.dump(best, model_path)
    logger.info("Model saved to : %s", model_path)

    if scaler is not None and scaler_path is not None:
        joblib.dump(scaler, scaler_path)
        logger.info("Scaler saved to: %s", scaler_path)

    # ---- Persist metadata -----------------------------------------------
    meta = {
        "model_type":    model_type,
        "best_params":   gs.best_params_,
        "val_metrics":   val_metrics,
        "train_metrics": train_metrics,
        "n_train":       int(len(X_train)),
        "n_dev":         int(len(X_dev)),
        "embed_dim":     int(X_train.shape[1]),
        "scale":         scale,
        "train_csv":     str(train_csv),
        "dev_csv":       str(dev_csv),
        "elapsed_s":     round(elapsed, 2),
    }
    meta_path = output_dir / f"{model_type}_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Meta saved to  : %s", meta_path)

    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a downstream classifier on pre-extracted embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--train-csv", type=Path, required=True,
        help="CSV produced by extract_features.py for the training split.",
    )
    p.add_argument(
        "--dev-csv", type=Path, required=True,
        help="CSV produced by extract_features.py for the development split.",
    )
    p.add_argument(
        "--model",
        choices=[
            "SVM", "LogisticRegression", "KNN",
            "NaiveBayes", "DecisionTree", "MLP",
            "XGBoost", "RandomForest",
        ],
        default="SVM",
        help="Downstream classifier to train.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("outputs/models"),
        help="Directory to write trained model artefacts.",
    )
    p.add_argument(
        "--no-scale", action="store_true",
        help="Disable StandardScaler preprocessing.",
    )
    p.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Parallel jobs for GridSearchCV (-1 = all cores).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    missing = []
    for attr in ("train_csv", "dev_csv"):
        path: Path = getattr(args, attr)
        if not path.exists():
            missing.append(str(path))
    if missing:
        logger.error("The following CSV files were not found:\n  %s", "\n  ".join(missing))
        sys.exit(1)

    train(
        train_csv=args.train_csv,
        dev_csv=args.dev_csv,
        model_type=args.model,
        output_dir=args.output_dir,
        scale=not args.no_scale,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
