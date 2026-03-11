"""Audio preprocessing pipeline for base64-encoded audio input."""

import base64
import io
import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
SILENCE_THRESHOLD_DB = 20


class AudioPreprocessingError(Exception):
    """Raised when audio preprocessing fails at any stage."""


def _decode_base64(audio_base64):
    """Decode a base64 string into raw audio bytes."""
    try:
        return base64.b64decode(audio_base64)
    except Exception as e:
        raise AudioPreprocessingError(
            f"Incorrect Base64 String: {e}"
        ) from e


def _load_audio(audio_bytes):
    """Load raw audio bytes into a waveform array."""
    try:
        audio_buffer = io.BytesIO(audio_bytes)
        return librosa.load(audio_buffer, sr=None, mono=False)
    except Exception as e:
        raise AudioPreprocessingError(
            f"Failed to load MP3 audio: {e}"
        ) from e


def _to_mono(waveform):
    """Convert multi-channel audio to mono."""
    if waveform.ndim > 1:
        return librosa.to_mono(waveform)
    return waveform


def _resample(waveform, current_sr, target_sr):
    """Resample waveform to the target sample rate if needed."""
    if current_sr != target_sr:
        return librosa.resample(
            waveform, orig_sr=current_sr, target_sr=target_sr
        )
    return waveform


def _trim_silence(waveform):
    """Remove leading and trailing silence from the waveform."""
    try:
        trimmed, _ = librosa.effects.trim(
            waveform, top_db=SILENCE_THRESHOLD_DB
        )
        return trimmed
    except Exception as e:
        raise AudioPreprocessingError(
            f"Failed to trim silence: {e}"
        ) from e


def _normalize(waveform):
    """Normalize the waveform amplitude to [-1, 1]."""
    try:
        return librosa.util.normalize(waveform)
    except Exception as e:
        raise AudioPreprocessingError(
            f"Failed to normalize amplitude: {e}"
        ) from e


def preprocess_audio(audio_base64, target_sample_rate=TARGET_SAMPLE_RATE):
    """Full preprocessing pipeline: decode, load, mono, resample, trim, normalize."""
    try:
        audio_bytes = _decode_base64(audio_base64)
        waveform, sample_rate = _load_audio(audio_bytes)
        waveform = _to_mono(waveform)
        waveform = _resample(waveform, sample_rate, target_sample_rate)
        waveform = waveform.astype(np.float32)
        waveform = _trim_silence(waveform)
        waveform = _normalize(waveform)
        return waveform, target_sample_rate
    except AudioPreprocessingError:
        raise
    except Exception as e:
        raise AudioPreprocessingError(
            f"Unexpected error during preprocessing: {e}"
        ) from e


def preprocess_audio_from_bytes(raw_bytes, target_sample_rate=TARGET_SAMPLE_RATE):
    """Preprocessing pipeline for raw audio bytes (from file upload)."""
    try:
        waveform, sample_rate = _load_audio(raw_bytes)
        waveform = _to_mono(waveform)
        waveform = _resample(waveform, sample_rate, target_sample_rate)
        waveform = waveform.astype(np.float32)
        waveform = _trim_silence(waveform)
        waveform = _normalize(waveform)
        return waveform, target_sample_rate
    except AudioPreprocessingError:
        raise
    except Exception as e:
        raise AudioPreprocessingError(
            f"Unexpected error during preprocessing: {e}"
        ) from e
