# ============================================================
# dataset_analyzer/models/audio_model.py
# Hybrid Audio Classifier with Lazy Loading and Robust Fallback
# ============================================================
from __future__ import annotations

import logging
import os
import warnings
from typing import List, Tuple, Union, Optional
import math

# --- 1. Gestion des Dépendances Optionnelles ---
_AUDIO_DEPS_AVAILABLE = False
try:
    import torch
    import torchaudio

    _AUDIO_DEPS_AVAILABLE = True
except ImportError:
    # Les fonctions internes gèreront le cas où torchaudio n'est pas là
    pass

logger = logging.getLogger("DataSense.AudioModel")

# Désactiver les avertissements courants liés à l'audio/torch
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------------------------------------
# Audio Analysis Service (Lazy Loading Core)
# -----------------------------------------------------------

class AudioClassifierService:
    """
    Manages the lazy loading and inference of the torchaudio pipeline,
    and provides audio stats fallback.
    """

    def __init__(self):
        self._model: Optional[torch.nn.Module] = None
        self._labels: List[str] = []
        self._is_loaded = False

        # Initialisation du device si les dépendances sont là, sinon CPU par défaut
        self._device = torch.device("cpu")
        if _AUDIO_DEPS_AVAILABLE:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._target_sr = 16000  # Standard sample rate for many pipelines

    def _try_load_pipeline(self) -> bool:
        """Loads the model and labels once via introspection of torchaudio.pipelines."""

        if not _AUDIO_DEPS_AVAILABLE:
            logger.warning(
                "Skipping Audio Analysis: PyTorch/Torchaudio not found. "
                "Install them (e.g., pip install torch torchaudio) to enable ML-based analysis."
            )
            return False

        if self._is_loaded:
            return True

        logger.info("Initializing Torchaudio pipeline (lazy load)...")

        try:
            import torchaudio.pipelines as pipelines

            # Common pipeline name substrings we care about (PANNs/SED/YAMNet)
            candidate_substrings = ("PANN", "SED", "YAMNET", "HEAR")

            # --- 1. Introspection : Recherche du pipeline
            for name in dir(pipelines):
                if name.startswith("_"):
                    continue
                name_upper = name.upper()
                if any(s in name_upper for s in candidate_substrings):
                    try:
                        pipeline = getattr(pipelines, name)
                        model = pipeline.get_model() if hasattr(pipeline, "get_model") else pipeline
                        labels = list(pipeline.labels) if hasattr(pipeline, "labels") else []

                        if model is not None and labels:  # Le modèle doit être utilisable
                            model.to(self._device)
                            model.eval()
                            logger.info("✅ Loaded audio pipeline: torchaudio.pipelines.%s", name)
                            self._model = model
                            self._labels = labels
                            self._is_loaded = True
                            return True
                    except Exception as e:
                        logger.debug("Failed loading torchaudio.pipelines.%s: %s", name, e)

            # --- 2. Échec de la recherche
            logger.warning(
                "⚠️ No compatible torchaudio pipeline found (PANNs/SED/YAMNet). "
                "Audio ML will be unavailable."
            )
            self._is_loaded = False
            return False

        except Exception as e:
            logger.error("Failed to load torchaudio pipelines: %s", e)
            self._is_loaded = False
            return False

    def _audio_stats_fallback(self, audio_path: str) -> str:
        """
        Compute safe audio stats (duration, sample rate, RMS, peak) and return a short string.
        Robust to missing dependencies by checking _AUDIO_DEPS_AVAILABLE.
        """
        if not _AUDIO_DEPS_AVAILABLE:
            return "Indeterminate (Dependencies missing)"

        try:
            # 1. Get info (preferred way)
            info = torchaudio.info(audio_path)
            sample_rate = int(info.sample_rate)
            num_frames = int(info.num_frames)
            duration = num_frames / sample_rate
        except Exception:
            # Fallback 2: load a small portion just to get SR and duration (less reliable)
            try:
                # Load small frame to avoid OOM for huge files
                waveform, sample_rate = torchaudio.load(audio_path, num_frames=1024)
                # We can't determine the true duration this way, so we skip it.
                duration = None
            except Exception as e:
                logger.debug("Fallback stat read failed for %s: %s", audio_path, e)
                return "Indeterminate"

        # 2. Compute RMS and peak using full load (requires memory but gives quality stats)
        rms = None
        peak = None
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.numel() > 0:
                # Convert to mono for stats
                waveform_mono = waveform.mean(dim=0).squeeze()
                rms = float(waveform_mono.abs().pow(2).mean().sqrt().item())
                peak = float(waveform_mono.abs().max().item())
        except Exception:
            # Ignore stats errors; duration/sr is usually the most important info
            pass

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

    def detect_audio_class(self, audio_path: str, top_k: int = 3) -> Union[str, List[Tuple[str, float]]]:

        if not self._try_load_pipeline():
            # Si le chargement ML échoue, utiliser les statistiques
            return self._audio_stats_fallback(audio_path)

        if not os.path.exists(audio_path):
            logger.warning("Audio file not found: %s", audio_path)
            return "Indeterminate"

        try:
            # 1. Chargement et Resample
            waveform, sr = torchaudio.load(audio_path)  # [channels, frames]

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # Mono

            # Resampling si nécessaire
            if sr != self._target_sr:
                # Seul le resample est nécessaire, pas la normalisation par maxv pour l'inférence
                waveform = torchaudio.functional.resample(waveform, sr, self._target_sr)

            waveform = waveform.to(self._device)

            # 2. Inférence
            with torch.inference_mode():
                outputs = self._model(waveform)

                # Gestion de la sortie [T, C] vs [C] ou tuple de sorties
                if isinstance(outputs, (list, tuple)):
                    scores = outputs[0]
                else:
                    scores = outputs

                if scores.dim() == 2:
                    avg_scores = scores.mean(dim=0)
                else:
                    avg_scores = scores.squeeze()

                probs = torch.nn.functional.softmax(avg_scores, dim=0)

                # 3. Top K
                k = min(top_k, len(self._labels)) if self._labels else top_k
                top_probs, top_idxs = torch.topk(probs, k=k)

                top_idxs_cpu = top_idxs.cpu().tolist()
                top_probs_cpu = top_probs.cpu().tolist()

                results: List[Tuple[str, float]] = []
                for idx, p in zip(top_idxs_cpu, top_probs_cpu):
                    # Protection contre les indices hors limites
                    label = self._labels[idx] if idx < len(self._labels) else f"class_{idx}"
                    results.append((label, float(p)))

                return results

        except Exception as exc:
            logger.exception("Audio classification failed for %s", audio_path)
            # fallback to stats when inference fails
            return self._audio_stats_fallback(audio_path)


# -----------------------------------------------------------
# Public API (Point d'entrée du module)
# -----------------------------------------------------------

_AUDIO_CLASSIFIER_SERVICE = AudioClassifierService()


def detect_audio_class(audio_path: str, top_k: int = 3) -> Union[str, List[Tuple[str, float]]]:
    """
    Public entry point for audio classification, using lazy loading service.

    Returns:
      - list[tuple[str, float]]: top-k label predictions (if ML is available)
      - str: "FallbackStats: ..." or "Indeterminate" otherwise
    """
    return _AUDIO_CLASSIFIER_SERVICE.detect_audio_class(audio_path, top_k)