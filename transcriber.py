"""Whisper-based multilingual audio transcription module.

Uses openai-whisper (tiny or base model) to automatically transcribe
audio waveforms to text. Supports all languages including Tamil, Hindi,
Malayalam, Telugu, and English.

Install:
    pip install openai-whisper
"""

import logging
import warnings
import numpy as np

logger = logging.getLogger(__name__)

# Suppress noisy FP16 CPU warnings from whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

_whisper_model = None


def _get_model():
    """Lazy-load Whisper model once and reuse on every request."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        # "base" is the best balance of speed vs accuracy for production.
        # Use "tiny" if latency is critical, "small" for higher accuracy.
        logger.info("Loading Whisper 'base' model...")
        _whisper_model = whisper.load_model("base")
        logger.info("Whisper model ready.")
    return _whisper_model


# Phase 1 supported languages (BCP-47 codes)
SUPPORTED_LANGS = {"en", "hi"}

# Languages that Whisper may confuse with Hindi
HINDI_ALIASES = {"ur", "sd"}  # Urdu, Sindhi (phonetically similar)


def transcribe(waveform: np.ndarray, sr: int, language: str = None) -> dict:
    """Transcribe a waveform to text using Whisper.

    Phase 1 supports English and Hindi.
    - English audio → English transcript (normal transcription)
    - Hindi audio → English transcript (Whisper's built-in translate feature)
    - Any other language → English transcript (translate)

    Args:
        waveform: 1D numpy float32 array at the sample rate used for loading.
        sr:       Sample rate of the waveform.
        language: Optional BCP-47 language code hint (e.g. "hi", "en").
                  If None, Whisper auto-detects the language.

    Returns:
        dict with keys:
            text       — full transcript string (always in English)
            language   — detected language code
            segments   — list of timed segments (start, end, text)
    """
    import whisper

    model = _get_model()

    # Whisper requires float32 waveform at 16kHz
    if sr != 16000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)

    waveform = waveform.astype(np.float32)

    try:
        # Step 1: Detect the language using Whisper
        # Load audio and get mel spectrogram for language detection
        audio = whisper.pad_or_trim(waveform)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)

        # Map Urdu/Sindhi to Hindi for display purposes
        display_lang = detected_lang
        if detected_lang in HINDI_ALIASES:
            display_lang = "hi"

        logger.info("Detected language: %s (display as: %s)", detected_lang, display_lang)

        # Step 2: Transcribe or translate based on detected language
        if detected_lang == "en":
            # English audio → transcribe normally
            result = model.transcribe(
                waveform,
                language="en",
                task="transcribe",
                fp16=False,
                verbose=False,
            )
        else:
            # Non-English audio (Hindi, Urdu, etc.) → translate to English
            result = model.transcribe(
                waveform,
                task="translate",  # Whisper built-in translation to English
                fp16=False,
                verbose=False,
            )

        logger.info(
            "Transcription complete — Language: %s | Task: %s | Length: %d chars",
            display_lang,
            "transcribe" if detected_lang == "en" else "translate",
            len(result.get("text", ""))
        )

        return {
            "text": result.get("text", "").strip(),
            "language": display_lang,
            "segments": [
                {"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in result.get("segments", [])
            ],
        }
    except Exception as e:
        logger.error("Whisper transcription failed: %s", e)
        return {"text": "", "language": "unknown", "segments": [], "error": str(e)}
