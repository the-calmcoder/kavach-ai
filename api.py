"""Flask API for AI-generated voice detection and scam analysis.

Provides endpoints for:
- /api/analyze       — Upload MP3, get full AI voice + scam analysis
- /api/voice-detection — Legacy JSON endpoint
- /api/fraud-analysis  — Text-only fraud analysis
- /api/health        — Health check for Azure App Service
"""

import base64
import logging
import os
import time
import uuid

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from audio_processing import preprocess_audio, preprocess_audio_from_bytes
from decision import DecisionOrchestrator
from explainability import ExplainabilityEngine
from fraud_engine import FraudIntentEngine
from model_core import AudioInferenceEngine
from transcriber import transcribe

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
app.json.sort_keys = False

# Enable CORS for frontend integration using environment variables
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
CORS(app, resources={r"/api/*": {"origins": allowed_origins if allowed_origins != ["*"] else "*"}})

# Centralized Rate Limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# ---------------------------------------------------------------------------
# Configuration — Azure Key Vault aware
# ---------------------------------------------------------------------------
# In Azure App Service, set these as Application Settings.
# For Azure Key Vault integration, use Key Vault References:
#   @Microsoft.KeyVault(SecretUri=https://<vault-name>.vault.azure.net/secrets/API-KEY)
# The App Service will automatically resolve them via Managed Identity.
API_KEY = os.environ.get("API_KEY")
if not API_KEY or API_KEY == "hackathon-secret-123":
    logger.warning("API_KEY is not securely set in the environment. Enforcing strict rate limits instead.")

# Phase 1 supported languages (Whisper auto-detects, we validate after)
PHASE1_LANGUAGES = {"en", "hi"}  # English & Hindi
LANGUAGE_DISPLAY = {"en": "English", "hi": "Hindi"}

# Legacy endpoint settings
SUPPORTED_LANGUAGES = {"Tamil", "English", "Hindi", "Malayalam", "Telugu"}
REQUIRED_AUDIO_FORMAT = "mp3"
REQUIRED_FIELDS = ["language", "audioFormat", "audioBase64"]

# Max upload size (10 MB)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# ---------------------------------------------------------------------------
# Engine initialization
# ---------------------------------------------------------------------------
logger.info("Loading engines...")
audio_inference_engine = AudioInferenceEngine()
explainability_engine = ExplainabilityEngine()
decision_orchestrator = DecisionOrchestrator()
fraud_intent_engine = FraudIntentEngine()
logger.info("Engines ready.")


# ===========================================================================
# Helper functions
# ===========================================================================

def _authenticate():
    """Validate the API key from request headers.
    Skips authentication if API_KEY is not securely configured.
    """
    if not API_KEY or API_KEY.strip() == "" or API_KEY == "hackathon-secret-123":
        return None  # No strict auth; rely on rate limiting
        
    api_key = request.headers.get("x-api-key") or request.headers.get("X-Api-Key")
    if not api_key or api_key != API_KEY:
        return jsonify({
            "error": "Unauthorized",
            "message": "Invalid or missing API key"
        }), 401
        
    return None


def _parse_request():
    """Parse and validate the JSON request body."""
    try:
        data = request.get_json()
        if not data:
            raise ValueError("Empty JSON")
        return data, None
    except (ValueError, TypeError):
        return None, (jsonify({
            "error": "Bad Request",
            "message": "Invalid JSON body"
        }), 400)


def _validate_fields(data):
    """Check that all required fields are present and valid (legacy endpoint)."""
    for field in REQUIRED_FIELDS:
        if field not in data:
            return jsonify({
                "error": "Bad Request",
                "message": f"Missing required field: {field}"
            }), 400

    if data["language"] not in SUPPORTED_LANGUAGES:
        return jsonify({
            "error": "Bad Request",
            "message": f"Unsupported language: {data['language']}"
        }), 400

    if data["audioFormat"] != REQUIRED_AUDIO_FORMAT:
        return jsonify({
            "error": "Bad Request",
            "message": f"audioFormat must be '{REQUIRED_AUDIO_FORMAT}'"
        }), 400

    return None


def _run_pipeline(audio_base64, language):
    """Execute the full pipeline: preprocess → HuBERT + Whisper → fraud analysis."""
    waveform, sample_rate = preprocess_audio(audio_base64)

    # HuBERT: classify voice as AI or Human
    ai_probability, _ = audio_inference_engine.infer(waveform, sample_rate)
    classification, confidence = decision_orchestrator.classify(ai_probability)
    explanation_text, _ = explainability_engine.explain(
        waveform, sample_rate, classification, confidence
    )

    # Whisper: auto-transcribe, then run fraud analysis
    transcription = transcribe(waveform, sample_rate, language=language)
    transcript_text = transcription.get("text", "")

    if transcript_text:
        fraud_result = fraud_intent_engine.analyze_intent(transcript_text)
    else:
        fraud_result = {"verdict": "UNKNOWN", "combined_confidence_score": 0.0, "reason": "Audio could not be transcribed."}

    return {
        "language": language,
        "voice_analysis": {
            "verdict": classification,
            "confidence_score": round(float(confidence), 4),
            "explanation": explanation_text,
        },
        "fraud_analysis": {
            "verdict": fraud_result.get("verdict", "UNKNOWN"),
            "combined_confidence_score": fraud_result.get("combined_confidence_score", 0.0),
            "reason": fraud_result.get("reason", ""),
        },
    }


def _run_full_pipeline(raw_audio_bytes):
    """Execute the full pipeline from raw audio bytes (new /api/analyze endpoint).

    Steps:
        1. Preprocess raw audio bytes → waveform
        2. HuBERT → AI/Human classification
        3. Whisper → transcription + auto language detection
        4. Fraud engine → scam analysis
        5. Build comprehensive result
    """
    start_time = time.time()

    # 1. Preprocess audio from raw bytes
    waveform, sample_rate = preprocess_audio_from_bytes(raw_audio_bytes)
    preprocess_time = time.time()

    # 2. HuBERT: classify voice as AI or Human
    ai_probability, _ = audio_inference_engine.infer(waveform, sample_rate)
    classification, confidence = decision_orchestrator.classify(ai_probability)
    explanation_text, _ = explainability_engine.explain(
        waveform, sample_rate, classification, confidence
    )
    inference_time = time.time()

    # 3. Whisper: auto-transcribe + detect language
    transcription = transcribe(waveform, sample_rate, language=None)  # None = auto-detect
    transcript_text = transcription.get("text", "")
    detected_lang = transcription.get("language", "unknown")
    segments = transcription.get("segments", [])
    transcribe_time = time.time()

    # Map language code to display name
    language_display = LANGUAGE_DISPLAY.get(detected_lang, detected_lang)

    # Check if detected language is in Phase 1 supported set
    language_supported = detected_lang in PHASE1_LANGUAGES
    if not language_supported:
        language_display = f"{language_display} (detected — full support coming soon)"

    # 4. Fraud analysis on transcript
    if transcript_text:
        fraud_result = fraud_intent_engine.analyze_intent(transcript_text)
    else:
        fraud_result = {
            "verdict": "UNKNOWN",
            "combined_confidence_score": 0.0,
            "reason": "Audio could not be transcribed — unable to analyze for scam intent."
        }
    fraud_time = time.time()

    # 5. Compute audio metadata
    audio_duration = len(waveform) / sample_rate if sample_rate > 0 else 0.0

    total_time = time.time() - start_time

    return {
        "status": "success",
        "request_id": str(uuid.uuid4()),
        "language_detected": language_display,
        "language_code": detected_lang,
        "voice_analysis": {
            "verdict": classification,
            "confidence_score": round(float(confidence), 4),
            "ai_probability": round(float(ai_probability), 4),
            "explanation": explanation_text,
        },
        "fraud_analysis": {
            "verdict": fraud_result.get("verdict", "UNKNOWN"),
            "confidence_score": fraud_result.get("combined_confidence_score", 0.0),
            "reason": fraud_result.get("reason", ""),
        },
        "transcript": transcript_text,
        "detailed_analysis": {
            "audio_duration_seconds": round(audio_duration, 2),
            "sample_rate": sample_rate,
            "ai_probability_raw": round(float(ai_probability), 6),
            "language_auto_detected": detected_lang,
            "transcript_segments": segments,
        },
        "processing_time": {
            "preprocessing_ms": round((preprocess_time - start_time) * 1000),
            "voice_classification_ms": round((inference_time - preprocess_time) * 1000),
            "transcription_ms": round((transcribe_time - inference_time) * 1000),
            "fraud_analysis_ms": round((fraud_time - transcribe_time) * 1000),
            "total_ms": round(total_time * 1000),
        },
    }


# ===========================================================================
# Routes
# ===========================================================================




@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint for Azure App Service probes."""
    return jsonify({
        "status": "healthy",
        "service": "VoiceGuard AI",
        "version": "1.0.0",
        "engines": {
            "audio_inference": "loaded",
            "explainability": "loaded",
            "fraud_detection": "loaded",
        }
    }), 200


@app.route("/api/analyze", methods=["POST"])
@limiter.limit("5 per minute")
def analyze():
    """Main endpoint — upload an MP3 file and get full analysis.
    
    Accepts: multipart/form-data with a 'file' field containing an MP3 file.
    Returns: JSON with voice AI/Human verdict, scam detection, transcript, and detailed analysis.
    """
    # Authenticate is bypassed here intentionally for the public web frontend.
    # Security is handled via rate-limiting instead.
    
    # Validate file upload
    if "file" not in request.files:
        return jsonify({
            "error": "Bad Request",
            "message": "No file uploaded. Please upload an MP3 file using the 'file' field."
        }), 400

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        return jsonify({
            "error": "Bad Request",
            "message": "No file selected."
        }), 400

    # Validate file extension
    filename = uploaded_file.filename.lower()
    if not filename.endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac')):
        return jsonify({
            "error": "Bad Request",
            "message": "Unsupported file format. Please upload an MP3, WAV, M4A, OGG, or FLAC file."
        }), 400

    try:
        # Read the raw bytes from the uploaded file
        raw_audio_bytes = uploaded_file.read()

        if len(raw_audio_bytes) == 0:
            return jsonify({
                "error": "Bad Request",
                "message": "Uploaded file is empty."
            }), 400

        # Run the full analysis pipeline
        result = _run_full_pipeline(raw_audio_bytes)
        return jsonify(result), 200

    except Exception as e:
        logger.error("Analysis pipeline failed", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "An internal error occurred during analysis."
        }), 500


@app.route("/api/voice-detection", methods=["POST"])
@limiter.limit("5 per minute")
def voice_detection():
    """Legacy endpoint for AI voice detection (base64 JSON input)."""
    auth_error = _authenticate()
    if auth_error:
        return auth_error

    data, parse_error = _parse_request()
    if parse_error:
        return parse_error

    validation_error = _validate_fields(data)
    if validation_error:
        return validation_error

    try:
        result = _run_pipeline(data["audioBase64"], data["language"])
        return jsonify(result), 200
    except Exception as e:
        logger.error("Pipeline failed", exc_info=True)
        return jsonify({"status": "error", "message": "An internal error occurred."}), 500


@app.route("/api/fraud-analysis", methods=["POST"])
@limiter.limit("10 per minute")
def fraud_analysis():
    """Dedicated endpoint for text-only fraud intent analysis."""
    auth_error = _authenticate()
    if auth_error:
        return auth_error

    data, parse_error = _parse_request()
    if parse_error:
        return parse_error

    transcript = data.get("transcript", "")
    if not transcript:
        return jsonify({"error": "Bad Request", "message": "Missing 'transcript' field."}), 400

    try:
        fraud_result = fraud_intent_engine.analyze_intent(transcript)
        return jsonify({
            "verdict": fraud_result.get("verdict", "UNKNOWN"),
            "combined_confidence_score": fraud_result.get("combined_confidence_score", 0.0),
            "reason": fraud_result.get("reason", ""),
        }), 200
    except Exception as e:
        logger.error("Fraud analysis failed", exc_info=True)
        return jsonify({"status": "error", "message": "An internal error occurred."}), 500


# ===========================================================================
# Frontend Serving
# ===========================================================================

@app.route("/")
def serve_frontend():
    """Serve the frontend website."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    """Serve static frontend assets (CSS, JS, images)."""
    return send_from_directory(FRONTEND_DIR, path)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
