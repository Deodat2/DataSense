# dataset_analyzer/audio_model.py
# Hybrid audio classifier (PRO)
#  - Try to use a torchaudio pretrained pipeline (PANNs/SED or others) when available.
#  - If no supported pipeline exists, gracefully fall back to a lightweight audio stats summary.
#  - Safe device handling, no automatic brittle downloads, clear logging.
from __future__ import annotations

import logging
import os
from typing import List, Tuple, Union, Optional

import torch
import torchaudio

logger = logging.getLogger(__name__)

# Device (GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _try_load_torchaudio_pipeline() -> Tuple[Optional[torch.nn.Module], List[str]]:
    """
    Try to discover a suitable pretrained audio pipeline in torchaudio.pipelines.

    Strategy:
      - import torchaudio.pipelines
      - look for common pipeline names or anything exposing `get_model` and `labels`
      - return (model, labels) or (None, [])
    """
    try:
        import torchaudio.pipelines as pipelines
    except Exception as e:
        logger.debug("torchaudio.pipelines unavailable: %s", e)
        return None, []

    # Common pipeline name substrings we care about (PANNs/SED/etc.)
    candidate_substrings = ("PANN", "SED", "YAMNET", "HEAR", "PANNs")

    for name in dir(pipelines):
        if name.startswith("_"):
            continue
        name_upper = name.upper()
        if any(s in name_upper for s in candidate_substrings):
            try:
                pipeline = getattr(pipelines, name)
                # pipeline could be an object exposing get_model() and labels
                model = None
                labels: List[str] = []

                if hasattr(pipeline, "get_model"):
                    model = pipeline.get_model()
                else:
                    # pipeline might already be a model or factory
                    model = pipeline

                if hasattr(pipeline, "labels"):
                    try:
                        labels = list(pipeline.labels)
                    except Exception:
                        labels = []

                # If model loaded, move to device and eval
                if model is not None:
                    try:
                        model.to(DEVICE)
                        model.eval()
                        logger.info("✅ Loaded audio pipeline: torchaudio.pipelines.%s", name)
                        return model, labels
                    except Exception as e:
                        logger.debug("Failed to prepare model from %s: %s", name, e)
                        # continue trying other candidates
            except Exception as e:
                logger.debug("Failed loading torchaudio.pipelines.%s: %s", name, e)

    # Try a last-ditch explicit known attribute access (some torchaudio versions expose constants)
    for explicit_name in ("PANNsSED", "PANN_SED", "PANN", "YAMNET"):
        pipeline = getattr(pipelines, explicit_name, None) if "pipelines" in locals() else None
        if pipeline is not None:
            try:
                model = pipeline.get_model() if hasattr(pipeline, "get_model") else pipeline
                labels = list(pipeline.labels) if hasattr(pipeline, "labels") else []
                model.to(DEVICE)
                model.eval()
                logger.info("✅ Loaded explicit pipeline: torchaudio.pipelines.%s", explicit_name)
                return model, labels
            except Exception as e:
                logger.debug("Explicit pipeline %s failed: %s", explicit_name, e)

    logger.warning(
        "⚠️ No compatible torchaudio pipeline found (PANNs/SED/YAMNet). "
        "Audio ML will be unavailable — falling back to safe audio stats."
    )
    return None, []


def _audio_stats_fallback(audio_path: str) -> str:
    """Compute safe audio stats (duration, sample rate, RMS, peak) and return a short string."""
    try:
        info = torchaudio.info(audio_path)
        # torchaudio.info returns an object, use available attributes
        sample_rate = int(info.sample_rate) if hasattr(info, "sample_rate") else None
        num_frames = int(info.num_frames) if hasattr(info, "num_frames") else None
    except Exception:
        # Last-resort: load small portion to compute stats (may be heavier)
        try:
            waveform, sample_rate = torchaudio.load(audio_path, num_frames=1024)
            num_frames = waveform.shape[1]
        except Exception as e:
            logger.debug("Fallback stat read failed for %s: %s", audio_path, e)
            return "Indeterminate"

    # compute duration if possible
    if sample_rate and num_frames:
        duration = num_frames / sample_rate
    else:
        duration = None

    # compute RMS and peak using full load (safe for small files)
    try:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.numel() == 0:
            return "Indeterminate"
        # convert to mono for stats
        if waveform.shape[0] > 1:
            waveform_mono = waveform.mean(dim=0)
        else:
            waveform_mono = waveform.squeeze(0)
        rms = float(waveform_mono.abs().pow(2).mean().sqrt().item())
        peak = float(waveform_mono.abs().max().item())
    except Exception:
        rms = None
        peak = None

    parts = []
    if duration is not None:
        parts.append(f"dur={duration:.2f}s")
    if sample_rate:
        parts.append(f"sr={sample_rate}")
    if rms is not None:
        parts.append(f"rms={rms:.4f}")
    if peak is not None:
        parts.append(f"peak={peak:.4f}")

    return "FallbackStats: " + ", ".join(parts) if parts else "Indeterminate"


# Attempt to load a model once at import time
_yamnet_model, _class_names = _try_load_torchaudio_pipeline()
# Keep names in module-scope variables for quicker access
_model = _yamnet_model
_class_names = _class_names


def detect_audio_class(audio_path: str, top_k: int = 3) -> Union[str, List[Tuple[str, float]]]:
    """
    Hybrid: if a pretrained torchaudio pipeline is available, return top-k (label, prob).
    Otherwise return a short informative string produced by the fallback stats function.

    Return types:
      - list[tuple[str, float]]: top-k label predictions (best-case)
      - str: "FallbackStats: ..." or "Indeterminate" when ML unavailable
    """
    # If we have no model, return safe fallback stats (string) rather than crashing
    if _model is None or not isinstance(_class_names, list):
        logger.debug("Audio model not present; using stats fallback for %s", audio_path)
        return _audio_stats_fallback(audio_path)

    if not os.path.exists(audio_path):
        logger.warning("Audio file not found: %s", audio_path)
        return "Indeterminate"

    try:
        waveform, sr = torchaudio.load(audio_path)  # [channels, frames]

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        target_sr = 16000  # many pipelines expect 16k, but we'll resample only if necessary
        if sr != target_sr:
            try:
                waveform = torchaudio.functional.resample(waveform, sr, target_sr)
                sr = target_sr
            except Exception:
                # if resample fails, continue with original sr
                logger.debug("Resample failed or not required for %s", audio_path)

        waveform = waveform.to(DEVICE)
        # safe normalization
        maxv = waveform.abs().max()
        if maxv > 0:
            waveform = waveform / maxv

        with torch.inference_mode():
            outputs = _model(waveform)

            # outputs may be:
            #  - a single tensor of scores (T x C or C)
            #  - tuple/list where first element is scores
            if isinstance(outputs, (list, tuple)):
                scores = outputs[0]
            else:
                scores = outputs

            # handle [T, C] temporal dimension vs [C]
            if scores.dim() == 2:
                avg_scores = scores.mean(dim=0)
            else:
                avg_scores = scores.squeeze()

            probs = torch.nn.functional.softmax(avg_scores, dim=0)

            k = min(top_k, len(_class_names)) if _class_names else top_k
            top_probs, top_idxs = torch.topk(probs, k=k)

            top_idxs_cpu = top_idxs.cpu().tolist()
            top_probs_cpu = top_probs.cpu().tolist()

            results: List[Tuple[str, float]] = []
            for idx, p in zip(top_idxs_cpu, top_probs_cpu):
                label = _class_names[idx] if idx < len(_class_names) else f"class_{idx}"
                results.append((label, float(p)))

            return results

    except Exception as exc:
        logger.exception("Audio classification failed for %s: %s", audio_path, exc)
        # fallback to stats when inference fails
        return _audio_stats_fallback(audio_path)


# CLI quick test when executed directly
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Hybrid audio classifier (dataset_analyzer)")
    p.add_argument("audio", help="Path to an audio file")
    p.add_argument("--topk", type=int, default=3, help="Top K predictions")
    args = p.parse_args()

    out = detect_audio_class(args.audio, top_k=args.topk)
    print(out)
