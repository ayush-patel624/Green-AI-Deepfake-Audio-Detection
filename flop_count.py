"""
flop_count.py
-------------
Compute Multiply-Accumulate (MAC) count and parameter count for Wav2Vec 2.0
models loaded via torchaudio, with custom hooks for Transformer sub-modules
that ptflops cannot handle automatically.

Author : Ayush Patel, IIIT Guwahati

Usage
-----
    python src/flop_count.py --model BASE --audio-duration 3.5
    python src/flop_count.py --model BASE --audio-file /path/to/audio.flac
    python src/flop_count.py --model LARGE --audio-duration 3.5
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torchaudio.utils import download_asset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry — torchaudio pipeline bundles
# ---------------------------------------------------------------------------
MODEL_BUNDLES = {
    "BASE":  torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H,
    "LARGE": torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H,
}

TARGET_SR = 16_000
DEFAULT_AUDIO_DURATION = 3.5    # seconds — matches paper's training set average


# ---------------------------------------------------------------------------
# ptflops custom hooks
# ---------------------------------------------------------------------------

def _layer_norm_hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
    """
    MACs for LayerNorm: each output element is computed from one mean and one
    std computation, both O(C) work, giving ~2*B*N*C total operations.
    We count it conservatively as B * N * C multiplications.
    """
    x = input[0]
    if x.dim() == 3:
        B, N, C = x.shape
    elif x.dim() == 2:
        B, N, C = 1, x.shape[0], x.shape[1]
    else:
        return
    module.__flops__ += int(B * N * C)


def _multihead_attention_hook(module: nn.Module, input: tuple, output) -> None:  # noqa: ANN001
    """
    Manual MAC count for nn.MultiheadAttention (or compatible interface):

      1. Linear projections Q, K, V: 3 * B * N * D * D  (in-proj is fused)
      2. Q @ K^T:                    B * heads * N * (D/heads) * N
      3. softmax (weights by V):     B * heads * N * N * (D/heads)
      4. out-projection:             B * N * D * D

    For modules where embed_dim is accessible via .embed_dim and .num_heads.
    """
    # input[0] is the query tensor — shape (N, B, D) or (B, N, D)
    q = input[0]
    if q.dim() == 3:
        dim0, dim1, D = q.shape
        # torchaudio models use (N, B, D) ordering
        N = dim0 if dim0 > dim1 else dim1
        B = dim1 if dim0 > dim1 else dim0
    else:
        return

    embed_dim = getattr(module, "embed_dim", D)
    num_heads  = getattr(module, "num_heads",  8)
    head_dim   = embed_dim // num_heads

    flops = 0
    # Q, K, V projections (in_proj_weight: [3*D, D])
    flops += 3 * B * N * embed_dim * embed_dim
    # Q @ K^T
    flops += B * num_heads * N * head_dim * N
    # Attn weights @ V
    flops += B * num_heads * N * N * head_dim
    # Output projection
    flops += B * N * embed_dim * embed_dim

    module.__flops__ += flops


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_waveform(
    audio_file: Path | None,
    duration: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns a (1, T) waveform tensor on *device*.
    If *audio_file* is None a synthetic zero tensor of the requested duration is used.
    """
    if audio_file is not None:
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        try:
            import soundfile as sf
            import numpy as np
            data, sr = sf.read(str(audio_file), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            waveform = torch.tensor(data).unsqueeze(0)
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                waveform = resampler(waveform)
        except ImportError:
            waveform, sr = torchaudio.load(str(audio_file))
            if sr != TARGET_SR:
                waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        logger.info("Loaded waveform: %s  shape=%s", audio_file.name, waveform.shape)
    else:
        n_samples = int(duration * TARGET_SR)
        waveform = torch.zeros(1, n_samples)
        logger.info(
            "Using synthetic waveform: %.2f s  (%d samples @ %d Hz)",
            duration, n_samples, TARGET_SR,
        )

    return waveform.to(device)


def count_flops(
    model_key: str = "BASE",
    audio_file: Path | None = None,
    audio_duration: float = DEFAULT_AUDIO_DURATION,
    layer_slice: int | None = None,
) -> None:
    """
    Compute and print MACs and parameter count.

    Parameters
    ----------
    model_key      : 'BASE' or 'LARGE'
    audio_file     : path to a .flac/.wav file; if None a synthetic signal is used
    audio_duration : duration in seconds for the synthetic signal
    layer_slice    : if given, only feature encoder + first *layer_slice*
                     transformer layers are used (Green AI partial model)
    """
    try:
        from ptflops import get_model_complexity_info  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ptflops is required. Install with:  pip install ptflops"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Load model -------------------------------------------------------
    if model_key not in MODEL_BUNDLES:
        raise ValueError(f"Unknown model key '{model_key}'. Choose from {list(MODEL_BUNDLES)}")

    bundle = MODEL_BUNDLES[model_key]
    model  = bundle.get_model().to(device).eval()
    logger.info("Loaded model: WAV2VEC2_ASR_%s_960H", model_key)

    # ---- Optionally slice to first N transformer layers ------------------
    if layer_slice is not None:
        logger.info("Slicing model to first %d transformer layers.", layer_slice)
        try:
            # torchaudio wav2vec2 model: model.encoder.transformer.layers
            original_layers = model.encoder.transformer.layers
            model.encoder.transformer.layers = nn.ModuleList(
                list(original_layers)[:layer_slice]
            )
            logger.info(
                "Original transformer layers: %d  →  sliced to: %d",
                len(original_layers), layer_slice,
            )
        except AttributeError:
            logger.warning(
                "Could not slice transformer layers automatically "
                "(model architecture may differ). Running full model."
            )

    # ---- Build input waveform --------------------------------------------
    waveform = load_waveform(audio_file, audio_duration, device)

    # ptflops expects a callable that returns the input tensor(s);
    # it passes a tuple of the shape you provide to input_constructor.
    def input_constructor(_ignored_shape):
        return waveform

    # ---- Custom hooks for Transformer modules ----------------------------
    custom_hooks: dict = {}

    # Collect all LayerNorm and MultiheadAttention instances in the model
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            custom_hooks[type(module)] = _layer_norm_hook
        if isinstance(module, nn.MultiheadAttention):
            custom_hooks[type(module)] = _multihead_attention_hook

    # ---- Run ptflops -----------------------------------------------------
    logger.info("Running ptflops … (this may take a moment)")

    macs_str, params_str = get_model_complexity_info(
        model,
        input_res=(1,),                     # dummy shape; overridden by input_constructor
        input_constructor=input_constructor,
        as_strings=True,
        print_per_layer_stat=True,
        verbose=False,
        custom_modules_hooks=custom_hooks if custom_hooks else None,
    )

    # ---- Print summary ---------------------------------------------------
    separator = "─" * 52
    print(f"\n{separator}")
    print(f"  FLOPs / MACs Analysis  —  WAV2VEC2 {model_key}")
    print(separator)
    print(f"  MACs (multiply-accumulate)  : {macs_str}")
    print(f"  Total parameters            : {params_str}")
    print(f"  Audio duration              : {audio_duration:.2f} s")
    if layer_slice is not None:
        print(f"  Transformer layers used     : {layer_slice}")
    print(separator + "\n")

    # ---- Count trainable params manually ---------------------------------
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total params     : %s", f"{total_params:,}")
    logger.info("Trainable params : %s", f"{trainable_params:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Count MACs and parameters for Wav2Vec 2.0 models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        choices=["BASE", "LARGE"],
        default="BASE",
        help="Wav2Vec 2.0 model variant.",
    )
    p.add_argument(
        "--audio-file", type=Path, default=None,
        help="Path to a .flac/.wav file to use as input. "
             "If omitted a synthetic signal is generated.",
    )
    p.add_argument(
        "--audio-duration", type=float, default=DEFAULT_AUDIO_DURATION,
        help="Duration (seconds) for synthetic input when --audio-file is not given.",
    )
    p.add_argument(
        "--layer-slice", type=int, default=None,
        help="Use only the first N transformer layers (Green AI partial model). "
             "E.g. --layer-slice 2 matches the best-performing configuration.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    count_flops(
        model_key=args.model,
        audio_file=args.audio_file,
        audio_duration=args.audio_duration,
        layer_slice=args.layer_slice,
    )


if __name__ == "__main__":
    main()
