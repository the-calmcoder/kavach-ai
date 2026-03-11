"""Enhanced Semantic Fraud Detection Engine.

Uses SentenceTransformer cosine similarity across 6 weighted detection
dimensions to identify scam patterns in call transcripts. If a trained
`fraud_weights.pkl` file exists, it overrides the default static weights
with mathematically calibrated per-dimension values.
"""

import os
import json
import logging
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

logger = logging.getLogger(__name__)

# Default static weights from the feature design table.
# Keys must match DIMENSIONS dict below. Feature #6 (Caller ID) is excluded.
DEFAULT_WEIGHTS = {
    "PRESSURE_URGENCY":     0.25,
    "TECHNICAL_MARKERS":    0.20,
    "FINANCIAL_RED_FLAGS":  0.20,
    "INSTITUTIONAL_DEVIATION": 0.15,
    "BEHAVIORAL_PERSUASION": 0.10,
    "ISOLATION_TACTICS":    0.10,  # Feature #7 (renormalized after dropping #6)
}

# Six semantic detection dimensions.
# Anchors are sourced directly from the provided scam detection dataset v2.0
# and merged with previously existing anchor phrases.
DIMENSIONS = {
    "PRESSURE_URGENCY": {
        "description": "High-pressure threats, legal fear, and artificial scarcity.",
        "anchors": [
            "Police are on their way",
            "Your identity has been used in a criminal case",
            "You are under digital arrest, do not leave the camera",
            "A parcel containing illegal substances was found in your name",
            "This is a recorded line from the Supreme Court",
            "Failure to comply will lead to immediate imprisonment",
            "Arrest warrant", "Deportation", "Criminal case", "Legal action",
            "Court summons", "Asset seizure", "Narcotics department", "Money laundering",
            "You have been selected for a lottery",
            "The offer is only valid while you are on this call",
            "Claim your government subsidy before the portal closes",
            "Your tax refund is ready, just pay the processing fee",
            "This is a pre-approved loan with 0% interest",
            "Limited-time offer", "Ending in 5 minutes",
            "Contract expires today", "Only one spot left",
            "Last chance to claim", "Government grant", "Lottery winner",
        ],
    },
    "TECHNICAL_MARKERS": {
        "description": "Remote access tools, IP threats, and device compromise.",
        "anchors": [
            "Your computer is sending error signals to our server",
            "I need you to download a support tool for a quick scan",
            "We have detected a security breach on your IP address",
            "Do not turn off your computer or you will lose all data",
            "I will provide you a 9-digit code to secure your connection",
            "Download this app", "Install the file", "Open AnyDesk",
            "Install TeamViewer", "Download remote access software",
            "Allow screen sharing", "Give me remote access",
            "Install this APK file", "Enable remote desktop",
            "Install Quick Support app", "Download the security update",
            "Virus detected", "System failure", "Remote access",
            "Security breach", "Your IP is public", "Malware infection",
            "RustDesk",
        ],
    },
    "FINANCIAL_RED_FLAGS": {
        "description": "OTP theft, QR scams, gift cards, and financial coercion.",
        "anchors": [
            "Share the OTP", "Tell me the code", "Verify password",
            "Read the SMS code", "Give me verification code",
            "Tell me the number you received", "What code did you get",
            "Type in your security digits", "Share your PIN with me",
            "Scan this QR code", "Open scanner and pay",
            "Receive money by scanning", "Use your payment app to scan",
            "Scan to verify your account", "GPay this QR code",
            "Your bank account will be blocked within 2 hours",
            "I am calling from the fraud prevention department",
            "A transaction of $999 is pending on your card",
            "To secure your funds, move them to our safe-haven account",
            "Verify your OTP to cancel the unauthorized charge",
            "Your KYC has expired, please update now to avoid deactivation",
            "Account suspended", "Unauthorized transaction", "Compromised",
            "Immediate payment", "Safety account", "KYC update",
            "Go to the nearest store and buy a physical gift card",
            "Read me the numbers on the back of the card",
            "We only accept payments via Bitcoin for security reasons",
            "The fee must be paid in cash through a courier service",
            "Gift card", "Apple voucher", "Bitcoin ATM",
            "Crypto wallet", "Wire transfer", "Western Union",
        ],
    },
    "INSTITUTIONAL_DEVIATION": {
        "description": "Refusal to use official channels or verify through the bank.",
        "anchors": [
            "Keep your phone in your pocket while you go to the bank",
            "The bank employees might be involved in the fraud, do not trust them",
            "Do not go to the bank branch",
            "You cannot verify this through the official number",
            "We are a special division, not listed publicly",
            "Do not use the number on the back of your card",
            "Our department does not have a public helpline",
            "Transfer funds to this special secure government account",
            "We will send a courier to collect the cash personally",
        ],
    },
    "BEHAVIORAL_PERSUASION": {
        "description": "Over-politeness, supervisor escalation, and love-bombing.",
        "anchors": [
            "I completely understand your concern and I am here to help",
            "You are absolutely right, let me escalate this to my supervisor",
            "Sir, I assure you this is completely legitimate",
            "We really value your cooperation in this sensitive matter",
            "My senior officer will personally handle your case",
            "I am going out of my way to help you today",
            "You are one of our most valued customers",
            "I am transferring you to a specialist line now",
            "Please hold, I am connecting you to the department head",
        ],
    },
    "ISOLATION_TACTICS": {
        "description": "Secrecy requests, threats if call ends, preventing outside help.",
        "anchors": [
            "This is a highly sensitive matter, do not discuss it with anyone",
            "If you hang up, we will be forced to take legal action",
            "Keep your phone in your pocket while you go to the bank",
            "Talking to family will compromise the investigation",
            "The bank employees might be involved in the fraud, do not trust them",
            "Do not hang up", "Stay on the line", "Keep this confidential",
            "Do not tell the bank", "Specialist line only",
            "Private investigation", "Gag order",
        ],
    },
}

SAFE_ANCHORS = [
    "I will not share my OTP",
    "I cannot give you that information",
    "This is just a song request",
    "Play some music",
    "I am merely an AI assistant",
    "Thank you for calling customer support",
    "Your order has been placed successfully",
    "Please hold while I transfer you",
    "I am calling to confirm your appointment",
    "How can I help you today",
    "Have a great day",
    "Your delivery is on the way",
]

# Risk thresholds
THRESHOLD_HIGH = 0.55
THRESHOLD_MEDIUM = 0.40


class FraudIntentEngine:
    """Semantic scam detection engine using weighted cosine-similarity dimensions."""

    def __init__(self, weights_path: str = "fraud_weights.pkl"):
        logger.info("Loading Semantic Fraud Engine...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load trained weights if available, otherwise use static defaults
        if os.path.exists(weights_path):
            try:
                self.weights = joblib.load(weights_path)
                logger.info("Loaded trained fraud weights from %s", weights_path)
            except Exception as e:
                logger.warning("Could not load %s: %s. Using defaults.", weights_path, e)
                self.weights = DEFAULT_WEIGHTS.copy()
        else:
            self.weights = DEFAULT_WEIGHTS.copy()
            logger.info("No trained weights found. Using default weights.")

        # Pre-compute anchor embeddings for all 6 dimensions
        self.dim_embeddings = {}
        for dim_key, dim_data in DIMENSIONS.items():
            self.dim_embeddings[dim_key] = self.model.encode(
                dim_data["anchors"], convert_to_tensor=True
            )

        self.safe_embeddings = self.model.encode(SAFE_ANCHORS, convert_to_tensor=True)
        logger.info("Fraud Engine ready with %d dimensions.", len(DIMENSIONS))

    def _score_sentence(self, sentence: str) -> dict:
        """Score a single sentence against all dimensions. Returns per-dim scores."""
        embedding = self.model.encode(sentence, convert_to_tensor=True)
        scores = {}
        for dim_key, anchor_vecs in self.dim_embeddings.items():
            cosine_scores = util.cos_sim(embedding, anchor_vecs)
            scores[dim_key] = float(torch.max(cosine_scores).item())

        safe_scores = util.cos_sim(embedding, self.safe_embeddings)
        scores["_safe"] = float(torch.max(safe_scores).item())
        return scores

    def analyze_intent(self, transcript: str) -> dict:
        """Analyze a call transcript and return a rich weighted fraud verdict.

        Returns:
            dict with keys: risk_level, weighted_score, top_dimension,
            dimension_scores, triggered_category, explanation
        """
        if not transcript or len(transcript.strip()) < 5:
            return {
                "risk_level": "UNKNOWN",
                "weighted_score": 0.0,
                "top_dimension": "NO_DATA",
                "dimension_scores": {},
                "triggered_category": "NO_DATA",
                "explanation": "No speech context available for analysis.",
            }

        sentences = [
            s.strip()
            for s in transcript.replace("!", ".").replace("?", ".").split(".")
            if len(s.strip()) >= 5
        ]

        # Aggregate the peak score per dimension across all sentences
        peak_scores = {dim: 0.0 for dim in DIMENSIONS}
        peak_safe = 0.0

        for sentence in sentences:
            scores = self._score_sentence(sentence)
            for dim in DIMENSIONS:
                if scores[dim] > peak_scores[dim]:
                    peak_scores[dim] = scores[dim]
            if scores["_safe"] > peak_safe:
                peak_safe = scores["_safe"]

        # Apply safe-context suppression:
        # If safe score is very high and beats the top fraud dim, reduce all scores
        top_fraud = max(peak_scores.values())
        if peak_safe > top_fraud:
            logger.debug("Safe context dominant (%.3f > %.3f). Suppressing.", peak_safe, top_fraud)
            suppression_factor = 1.0 - (peak_safe - top_fraud)
            peak_scores = {k: v * max(suppression_factor, 0.0) for k, v in peak_scores.items()}

        # Compute the final weighted score
        weighted_score = sum(
            peak_scores[dim] * self.weights.get(dim, 0.0)
            for dim in DIMENSIONS
        )

        # Find the most triggered dimension
        top_dimension = max(peak_scores, key=lambda d: peak_scores[d])
        top_dim_score = peak_scores[top_dimension]

        # Map to triggered category from dataset
        triggered_category = DIMENSIONS[top_dimension]["description"] if top_dim_score > THRESHOLD_MEDIUM else "NONE"

        # Assign binary verdict
        if weighted_score >= THRESHOLD_HIGH:
            verdict = "SCAM"
        elif weighted_score >= THRESHOLD_MEDIUM:
            # Medium range — lean SCAM, lower confidence
            verdict = "SCAM"
        else:
            verdict = "SAFE"

        # Confidence: distance from the neutral mid-point (0.475)
        MIDPOINT = (THRESHOLD_HIGH + THRESHOLD_MEDIUM) / 2
        if verdict == "SCAM":
            confidence = min(weighted_score / THRESHOLD_HIGH, 1.0)
        else:
            confidence = min((1.0 - weighted_score) / (1.0 - THRESHOLD_MEDIUM), 1.0)

        # One-liner reason
        if verdict == "SCAM":
            reason = f"Strong match on '{DIMENSIONS[top_dimension]['description']}'."
        else:
            reason = "No significant scam patterns detected."

        return {
            "verdict": verdict,
            "combined_confidence_score": round(float(confidence), 4),
            "reason": reason,
        }