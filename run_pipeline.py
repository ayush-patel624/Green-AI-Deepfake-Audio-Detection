"""
run_pipeline.py
---------------
End-to-end convenience script that chains feature extraction → training →
evaluation in a single CLI call.

Author : Ayush Patel, IIIT Guwahati

Example
-------
    python run_pipeline.py \
        --data-root /data/ASVspoof2019/LA \
        --output-dir outputs \
        --layer-index 2 \
        --model SVM \
        --device cpu
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---- Canonical ASVspoof 2019 LA directory layout -------------------------
#
# <data-root>/
# ├── ASVspoof2019_LA_train/flac/
# ├── ASVspoof2019_LA_dev/flac/
# ├── ASVspoof2019_LA_eval/flac/
# └── ASVspoof2019_LA_cm_protocols/
#     ├── ASVspoof2019.LA.cm.train.trn.txt
#     ├── ASVspoof2019.LA.cm.dev.trl.txt
#     └── ASVspoof2019.LA.cm.eval.trl.txt

SPLIT_CONFIG = {
    "train": {
        "audio_subdir":    "ASVspoof2019_LA_train/flac",
        "protocol_suffix": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    },
    "dev": {
        "audio_subdir":    "ASVspoof2019_LA_dev/flac",
        "protocol_suffix": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    },
    "eval": {
        "audio_subdir":    "ASVspoof2019_LA_eval/flac",
        "protocol_suffix": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
    },
}


def run_extraction(args, split_name: str, embed_dir: Path) -> Path:
    """Call extract_features.main() programmatically and return the output CSV path."""
    from extract_features import extract  # noqa: PLC0415

    cfg     = SPLIT_CONFIG[split_name]
    audio_dir = args.data_root / cfg["audio_subdir"]
    protocol  = args.data_root / cfg["protocol_suffix"]
    out_csv   = embed_dir / f"{split_name}_layer{args.layer_index}.csv"

    if out_csv.exists() and not args.force_reextract:
        logger.info("Embeddings already exist, skipping extraction: %s", out_csv)
        return out_csv

    if not audio_dir.exists():
        logger.error("Audio directory not found: %s", audio_dir)
        sys.exit(1)
    if not protocol.exists():
        logger.error("Protocol file not found: %s", protocol)
        sys.exit(1)

    logger.info("=== Extracting %s embeddings ===", split_name.upper())
    extract(
        audio_dir=audio_dir,
        protocol_file=protocol,
        output_csv=out_csv,
        layer_index=args.layer_index,
        device=args.device,
        max_files=args.max_files,
    )
    return out_csv


def run_training(args, train_csv: Path, dev_csv: Path, model_dir: Path) -> tuple[Path, Path]:
    """Call train.train() programmatically. Returns (model_path, scaler_path)."""
    from train import train  # noqa: PLC0415

    logger.info("=== Training %s ===", args.model)
    meta = train(
        train_csv=train_csv,
        dev_csv=dev_csv,
        model_type=args.model,
        output_dir=model_dir,
        scale=not args.no_scale,
        n_jobs=args.n_jobs,
    )

    model_path  = model_dir / f"{args.model}_model.pkl"
    scaler_path = model_dir / f"{args.model}_scaler.pkl" if not args.no_scale else None
    return model_path, scaler_path


def run_evaluation(args, eval_csv: Path, model_path: Path, scaler_path: Path | None, results_dir: Path) -> None:
    """Call evaluate.evaluate() programmatically."""
    from evaluate import evaluate  # noqa: PLC0415

    logger.info("=== Evaluating on test split ===")
    evaluate(
        test_csv=eval_csv,
        model_path=model_path,
        scaler_path=scaler_path,
        output_dir=results_dir,
        save_plots=not args.no_plots,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="End-to-end audio deepfake detection pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-root", type=Path, required=True,
        help="Root of the ASVspoof 2019 LA dataset.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("outputs"),
        help="Root directory for all outputs (embeddings, models, results).",
    )
    p.add_argument(
        "--layer-index", type=int, default=2,
        help="Wav2Vec 2.0 layer to use (0=feature encoder, 1-12=transformer).",
    )
    p.add_argument(
        "--model",
        choices=[
            "SVM", "LogisticRegression", "KNN",
            "NaiveBayes", "DecisionTree", "MLP",
            "XGBoost", "RandomForest",
        ],
        default="SVM",
        help="Downstream classifier.",
    )
    p.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu",
        help="Device for feature extraction.",
    )
    p.add_argument(
        "--no-scale", action="store_true",
        help="Skip StandardScaler on embeddings.",
    )
    p.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Parallel workers for GridSearchCV.",
    )
    p.add_argument(
        "--max-files", type=int, default=-1,
        help="Limit files per split during extraction (for quick testing; -1 = no limit).",
    )
    p.add_argument(
        "--force-reextract", action="store_true",
        help="Re-run feature extraction even if CSV already exists.",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip ROC/DET plot generation.",
    )
    p.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip extraction step (requires existing CSV files in --output-dir/embeddings).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable — falling back to CPU.")
        args.device = "cpu"

    embed_dir   = args.output_dir / "embeddings"
    model_dir   = args.output_dir / "models"
    results_dir = args.output_dir / "results"

    t_start = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Stage 1: Feature Extraction                                          #
    # ------------------------------------------------------------------ #
    if args.skip_extraction:
        train_csv = embed_dir / f"train_layer{args.layer_index}.csv"
        dev_csv   = embed_dir / f"dev_layer{args.layer_index}.csv"
        eval_csv  = embed_dir / f"eval_layer{args.layer_index}.csv"
        for p in (train_csv, dev_csv, eval_csv):
            if not p.exists():
                logger.error("Expected embedding CSV not found: %s", p)
                sys.exit(1)
    else:
        train_csv = run_extraction(args, "train", embed_dir)
        dev_csv   = run_extraction(args, "dev",   embed_dir)
        eval_csv  = run_extraction(args, "eval",  embed_dir)

    # ------------------------------------------------------------------ #
    # Stage 2: Training                                                    #
    # ------------------------------------------------------------------ #
    model_path, scaler_path = run_training(args, train_csv, dev_csv, model_dir)

    # ------------------------------------------------------------------ #
    # Stage 3: Evaluation                                                  #
    # ------------------------------------------------------------------ #
    run_evaluation(args, eval_csv, model_path, scaler_path, results_dir)

    elapsed = time.perf_counter() - t_start
    logger.info("Pipeline complete in %.1f s.", elapsed)


if __name__ == "__main__":
    main()