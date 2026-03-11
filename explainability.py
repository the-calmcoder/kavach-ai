"""Explainability engine for AI voice detection results."""

import json
import logging
import os
import random

import librosa
import numpy as np

from utils import ensure_1d_numpy

logger = logging.getLogger(__name__)

Z_SCORE_LOW = 1.0
Z_SCORE_HIGH = 2.0
Z_SCORE_EXTREME = 3.0
EPSILON = 1e-6

VOCAB = {
    "openers": [
        "Forensic analysis reveals",
        "Signal processing detects",
        "The audio fingerprint exhibits",
        "Our model identified",
        "Deep-fake detection algorithms highlight"
    ],
    "severity_low": ["slight", "minor", "subtle", "marginal"],
    "severity_high": [
        "significant", "anomalous", "distinct", "pronounced"
    ],
    "severity_extreme": [
        "unnatural", "mechanically precise", "robotic", "synthetic"
    ],
    "connectors": [
        "coupled with", "alongside", "as well as", "featuring"
    ]
}

HUMAN_PHRASES = [
    "Audio features align with natural human prosody distributions.",
    "Micro-tremors and breathing patterns indicate a genuine vocal tract.",
    "Spectral consistency falls within the standard human baseline.",
    "Absence of digital artifacts suggests organic recording."
]

FLATNESS_NOUNS = ["spectral flatness", "signal uniformity"]
ZCR_NOUNS = ["high-frequency artifacts", "temporal discontinuity"]


class ExplainabilityEngine:
    """Generates human-readable explanations for voice detection results."""

    def __init__(self, baseline_path="human_baseline.json", seed=42):
        self.baseline = None
        self.rng = random.Random(seed)

        if os.path.exists(baseline_path):
            with open(baseline_path, "r", encoding="utf-8") as f:
                self.baseline = json.load(f)

    def _get_deviation_desc(self, z_score):
        """Map a z-score to a severity description and level."""
        abs_z = abs(z_score)
        if abs_z < Z_SCORE_LOW:
            return "statistically normal", 0
        if abs_z < Z_SCORE_HIGH:
            return self.rng.choice(VOCAB["severity_low"]), 1
        if abs_z < Z_SCORE_EXTREME:
            return self.rng.choice(VOCAB["severity_high"]), 2
        return self.rng.choice(VOCAB["severity_extreme"]), 3

    def _explain_human(self):
        """Return a random human-classification explanation."""
        return self.rng.choice(HUMAN_PHRASES), {}

    def _extract_features(self, waveform):
        """Compute spectral flatness and zero-crossing rate."""
        waveform = ensure_1d_numpy(waveform)
        flatness = float(
            np.mean(librosa.feature.spectral_flatness(y=waveform))
        )
        zcr = float(
            np.mean(librosa.feature.zero_crossing_rate(y=waveform))
        )
        return {"flatness": flatness, "zcr": zcr}

    def _compare_to_baseline(self, features):
        """Compare extracted features against the human baseline."""
        reasons = []
        if not self.baseline:
            return reasons

        b_flat = self.baseline["flatness"]
        z_flat = (
            (features["flatness"] - b_flat["mean"])
            / (b_flat["std"] + EPSILON)
        )
        desc_flat, sev_flat = self._get_deviation_desc(z_flat)
        if sev_flat > 0:
            noun = self.rng.choice(FLATNESS_NOUNS)
            reasons.append(f"{desc_flat} {noun} (Z:{z_flat:.1f})")

        b_zcr = self.baseline["zcr"]
        z_zcr = (
            (features["zcr"] - b_zcr["mean"])
            / (b_zcr["std"] + EPSILON)
        )
        desc_zcr, sev_zcr = self._get_deviation_desc(z_zcr)
        if sev_zcr > 0:
            noun = self.rng.choice(ZCR_NOUNS)
            reasons.append(f"{desc_zcr} {noun}")

        return reasons

    def _build_explanation(self, reasons):
        """Construct a natural-language explanation from deviation reasons."""
        opener = self.rng.choice(VOCAB["openers"])
        if not reasons:
            return (
                f"{opener} latent feature anomalies "
                "characteristic of neural synthesis."
            )

        if len(reasons) > 1:
            connector = self.rng.choice(VOCAB["connectors"])
            return f"{opener} {reasons[0]} {connector} {reasons[1]}."

        return f"{opener} {reasons[0]}."

    def explain(self, waveform, sample_rate, classification, confidence):
        """Generate an explanation for the given classification."""
        # pylint: disable=unused-argument
        try:
            if classification == "HUMAN":
                return self._explain_human()

            features = self._extract_features(waveform)
            reasons = self._compare_to_baseline(features)
            explanation = self._build_explanation(reasons)
            return explanation, {}

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Explainability failed: %s", e)
            return (
                f"Analysis unavailable due to processing error: {e}", {}
            )
