"""
extract_features.py
-------------------
Extracts Wav2Vec 2.0 frame-level embeddings from a directory of .flac audio
files using the ASVspoof 2019 LA protocol label file structure.

Author : Ayush Patel, IIIT Guwahati
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2Model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "facebook/wav2vec2-base-960h"
EMBED_DIM = 768          # wav2vec2-base hidden size
TARGET_SR = 16_000       # wav2vec2 expects 16 kHz input


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_protocol(protocol_path: Path) -> pd.DataFrame:
    """
    Parse an ASVspoof 2019 LA protocol file into a DataFrame.

    Expected whitespace-separated columns:
        speaker_id  file_id  env  attack_type  label

    Returns a DataFrame with columns ['file_id', 'label'].
    """
    df = pd.read_csv(
        protocol_path,
        sep=r"\s+",
        header=None,
        names=["speaker_id", "file_id", "env", "attack_type", "label"],
        engine="python",
    )
    return df[["file_id", "label"]]


def read_audio(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """
    Read a .flac (or .wav) file, convert to mono float32, resample if needed.
    Returns a 1-D numpy array at *target_sr*.
    """
    data, sr = sf.read(str(path), dtype="float32")

    if data.ndim > 1:          # multi-channel → mono
        data = data.mean(axis=1)

    if sr != target_sr:
        try:
            import torchaudio
            waveform = torch.tensor(data).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            data = resampler(waveform).squeeze(0).numpy()
        except ImportError:
            logger.warning(
                "torchaudio not available for resampling; file %s has SR=%d, expected %d",
                path.name, sr, target_sr,
            )

    return data


# ---------------------------------------------------------------------------
# Embedding model wrapper
# ---------------------------------------------------------------------------

class Wav2Vec2Embedder:
    """
    Wraps facebook/wav2vec2-base-960h (or any compatible checkpoint) for
    frame-level feature extraction.

    Parameters
    ----------
    layer_index : int
        0  → feature-encoder output (before transformers)
        1-12 → transformer layer outputs (1-indexed; 12 is last)
    device : str
        'cpu' or 'cuda'
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        layer_index: int = 2,
        device: str = "cpu",
    ) -> None:
        self.layer_index = layer_index
        self.device = torch.device(device)

        logger.info("Loading model  : %s", model_id)
        logger.info("Layer index    : %d  (0=feature encoder, 1-12=transformer)", layer_index)

        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2Model.from_pretrained(
            model_id,
            output_hidden_states=True,
        )
        self.model.eval()
        self.model.to(self.device)

        n_transformer = self.model.config.num_hidden_layers   # 12 for base
        max_idx = n_transformer + 1   # +1 for feature-encoder (index 0)
        if not (0 <= layer_index <= n_transformer):
            raise ValueError(
                f"layer_index must be in [0, {n_transformer}]; got {layer_index}."
            )
        logger.info(
            "Model loaded. Transformer layers: %d. Using layer %d / %d.",
            n_transformer,
            layer_index,
            max_idx - 1,
        )

    @torch.no_grad()
    def embed(self, waveform: np.ndarray) -> np.ndarray:
        """
        Returns a mean-pooled 768-dim embedding (numpy float32).

        hidden_states[0]  = feature encoder output
        hidden_states[1]  = after transformer layer 1
        ...
        hidden_states[12] = after transformer layer 12
        """
        inputs = self.processor(
            waveform,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self.device)

        outputs = self.model(input_values)
        # outputs.hidden_states is a tuple of (n_transformer+2) tensors:
        # index 0 : feature extractor output
        # index 1 : after 1st transformer
        # ...
        hidden = outputs.hidden_states[self.layer_index]   # (1, T, 768)
        pooled = hidden.squeeze(0).mean(dim=0)             # (768,)
        return pooled.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract(
    audio_dir: Path,
    protocol_file: Path,
    output_csv: Path,
    layer_index: int = 2,
    device: str = "cpu",
    model_id: str = MODEL_ID,
    max_files: int = -1,
) -> None:
    """
    Iterate over the protocol file, locate matching .flac files in *audio_dir*,
    compute embeddings and write a CSV ready for downstream training.

    CSV columns:
        file_id, feature_0 … feature_767, label
    """
    protocol = load_protocol(protocol_file)
    logger.info("Protocol entries : %d", len(protocol))

    embedder = Wav2Vec2Embedder(
        model_id=model_id,
        layer_index=layer_index,
        device=device,
    )

    records: list[dict] = []
    skipped = 0

    iterable = protocol.iterrows()
    if max_files > 0:
        protocol = protocol.head(max_files)
        iterable = protocol.iterrows()

    for _, row in tqdm(iterable, total=len(protocol), desc="Extracting", unit="file"):
        file_id: str = row["file_id"]
        label: str   = row["label"]

        # Support both bare id and id.flac in the protocol
        stem = file_id if not file_id.endswith(".flac") else file_id[:-5]
        audio_path = audio_dir / f"{stem}.flac"

        if not audio_path.exists():
            logger.debug("File not found, skipping: %s", audio_path)
            skipped += 1
            continue

        try:
            waveform = read_audio(audio_path)
            embedding = embedder.embed(waveform)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to process %s: %s", audio_path.name, exc)
            skipped += 1
            continue

        record = {"file_id": stem}
        for i, v in enumerate(embedding):
            record[f"feature_{i}"] = v
        record["label"] = label
        records.append(record)

    logger.info(
        "Processed %d files, skipped %d.", len(records), skipped
    )

    if not records:
        logger.error("No embeddings were extracted. Check paths and protocol file.")
        sys.exit(1)

    df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Embeddings saved to : %s  (shape %s)", output_csv, df.shape)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract Wav2Vec 2.0 embeddings from ASVspoof .flac files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory containing .flac audio files (e.g. ASVspoof2019_LA_train/flac).",
    )
    p.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="ASVspoof 2019 LA protocol .txt file.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path (e.g. embeddings/train_layer2.csv).",
    )
    p.add_argument(
        "--layer-index",
        type=int,
        default=2,
        help="Wav2Vec 2.0 layer to extract from (0=feature encoder, 1-12=transformer).",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Compute device.",
    )
    p.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="HuggingFace model identifier.",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=-1,
        help="Limit number of files processed (useful for quick tests; -1 = no limit).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU.")
        args.device = "cpu"

    extract(
        audio_dir=args.audio_dir,
        protocol_file=args.protocol,
        output_csv=args.output,
        layer_index=args.layer_index,
        device=args.device,
        model_id=args.model_id,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
