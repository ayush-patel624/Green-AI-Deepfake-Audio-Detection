"""
evaluate.py
-----------
Evaluate a trained downstream classifier on the ASVspoof test split.
Reports Accuracy, Precision, Recall, F1, and Equal Error Rate (EER).
Optionally saves an ROC / DET curve PNG.

Author : Ayush Patel, IIIT Guwahati
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for headless servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label encoding (must match train.py)
# ---------------------------------------------------------------------------
LABEL_MAP = {"bonafide": 1, "spoof": 0}


# ---------------------------------------------------------------------------
# EER helpers
# ---------------------------------------------------------------------------

def compute_eer_speechbrain(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    """
    Compute EER using SpeechBrain's metric utility.

    Returns (eer_value, threshold).
    """
    try:
        from speechbrain.utils.metric_stats import EER as sb_EER  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "speechbrain is required for EER computation. "
            "Install it with:  pip install speechbrain"
        ) from exc

    y_true_t   = torch.tensor(y_true,   dtype=torch.float32)
    y_scores_t = torch.tensor(y_scores, dtype=torch.float32)

    positive_scores = y_scores_t[y_true_t == 1]
    negative_scores = y_scores_t[y_true_t != 1]

    eer_val, threshold = sb_EER(positive_scores, negative_scores)
    return float(eer_val), float(threshold)


def compute_eer_sklearn(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Fallback EER computation via scikit-learn's roc_curve.

    Returns (eer_value, threshold, fpr, fnr).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer_val  = float(fpr[eer_idx])
    threshold = float(thresholds[eer_idx])
    return eer_val, threshold, fpr, fnr


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_split(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    drop_cols = [c for c in ["file_id", "file_name"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    df["label"] = df["label"].map(LABEL_MAP)
    if df["label"].isna().any():
        raise ValueError("Unexpected label values in test CSV.")

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df["label"].values.astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_det_curve(fpr: np.ndarray, fnr: np.ndarray, eer: float, save_path: Path) -> None:
    """Save a Detection Error Tradeoff (DET) curve to PNG."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr * 100, fnr * 100, color="steelblue", linewidth=2, label="DET curve")
    ax.scatter(
        [eer * 100], [eer * 100],
        color="crimson", zorder=5, s=80,
        label=f"EER = {eer * 100:.2f}%",
    )
    ax.set_xlabel("False Acceptance Rate (%)", fontsize=12)
    ax.set_ylabel("False Rejection Rate (%)",  fontsize=12)
    ax.set_title("Detection Error Tradeoff (DET) Curve", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("DET curve saved to: %s", save_path)


def save_roc_curve(fpr: np.ndarray, fnr: np.ndarray, save_path: Path) -> None:
    """Save an ROC curve to PNG."""
    tpr = 1.0 - fnr
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2, label="ROC curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curve saved to : %s", save_path)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    test_csv: Path,
    model_path: Path,
    scaler_path: Path | None,
    output_dir: Path,
    save_plots: bool = True,
) -> dict:
    """
    Load a trained model, run inference on the test set, compute and
    display all metrics.

    Returns a metrics dict.
    """
    logger.info("Loading test split  : %s", test_csv)
    X_test, y_test = load_test_split(test_csv)
    logger.info("Test shape          : %s", X_test.shape)

    # ---- Load scaler (optional) -----------------------------------------
    if scaler_path is not None and scaler_path.exists():
        scaler = joblib.load(scaler_path)
        X_test = scaler.transform(X_test)
        logger.info("Applied scaler from : %s", scaler_path)
    elif scaler_path is not None:
        logger.warning(
            "Scaler path specified (%s) but file not found — skipping scaling.", scaler_path
        )

    # ---- Load model -------------------------------------------------------
    logger.info("Loading model from  : %s", model_path)
    model = joblib.load(model_path)
    model_name = type(model).__name__

    # ---- Inference --------------------------------------------------------
    y_pred   = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    # ---- Classification metrics ------------------------------------------
    acc  = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec  = float(recall_score(y_test, y_pred,    zero_division=0))
    f1   = float(f1_score(y_test, y_pred,        zero_division=0))

    # ---- EER (SpeechBrain) + fallback -----------------------------------
    try:
        eer_sb, eer_thresh_sb = compute_eer_speechbrain(y_test, y_scores)
        eer_source = "speechbrain"
    except Exception as exc:  # noqa: BLE001
        logger.warning("SpeechBrain EER failed (%s) — using sklearn fallback.", exc)
        eer_sb, eer_thresh_sb, _, _ = compute_eer_sklearn(y_test, y_scores)
        eer_source = "sklearn-fallback"

    # ---- sklearn EER for curve data -------------------------------------
    eer_sk, eer_thresh_sk, fpr, fnr = compute_eer_sklearn(y_test, y_scores)

    # ---- Print summary ---------------------------------------------------
    separator = "─" * 52
    print(f"\n{separator}")
    print(f"  Evaluation Results  —  {model_name}")
    print(separator)
    print(f"  Accuracy     : {acc * 100:7.4f} %")
    print(f"  Precision    : {prec * 100:7.4f} %")
    print(f"  Recall       : {rec * 100:7.4f} %")
    print(f"  F1 Score     : {f1 * 100:7.4f} %")
    print(separator)
    print(f"  EER ({eer_source:>17s}) : {eer_sb * 100:6.4f} %  @ θ={eer_thresh_sb:.6f}")
    print(separator + "\n")

    # ---- Plots ------------------------------------------------------------
    if save_plots:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_det_curve(fpr, fnr, eer_sk, output_dir / f"{model_name}_det.png")
        save_roc_curve(fpr, fnr,         output_dir / f"{model_name}_roc.png")

    # ---- Persist results JSON --------------------------------------------
    results = {
        "model":      model_name,
        "n_test":     int(len(y_test)),
        "accuracy":   round(acc,      6),
        "precision":  round(prec,     6),
        "recall":     round(rec,      6),
        "f1":         round(f1,       6),
        "eer":        round(eer_sb,   6),
        "eer_pct":    round(eer_sb * 100, 4),
        "eer_thresh": round(eer_thresh_sb, 8),
        "eer_source": eer_source,
        "test_csv":   str(test_csv),
        "model_path": str(model_path),
    }

    results_path = output_dir / f"{model_name}_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Results saved to    : %s", results_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained classifier on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--test-csv", type=Path, required=True,
        help="CSV produced by extract_features.py for the evaluation split.",
    )
    p.add_argument(
        "--model-path", type=Path, required=True,
        help="Path to the .pkl model file written by train.py.",
    )
    p.add_argument(
        "--scaler-path", type=Path, default=None,
        help="Path to the .pkl scaler file written by train.py (omit if --no-scale was used).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("outputs/results"),
        help="Directory for JSON results and plot PNGs.",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip saving ROC / DET curve plots.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    missing = []
    for p in [args.test_csv, args.model_path]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        logger.error("File(s) not found:\n  %s", "\n  ".join(missing))
        sys.exit(1)

    evaluate(
        test_csv=args.test_csv,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
