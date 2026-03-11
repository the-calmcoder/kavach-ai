"""Decision orchestration for AI voice classification."""

import logging

logger = logging.getLogger(__name__)

DECISION_THRESHOLD = 0.5


class DecisionOrchestrationError(Exception):
    """Raised when decision orchestration fails."""


class DecisionOrchestrator:
    """Classifies audio as AI-generated or human based on probability."""

    def __init__(self, decision_threshold=DECISION_THRESHOLD):
        self.decision_threshold = decision_threshold

    def classify(self, ai_probability):
        """Return (classification, confidence) from an AI probability score."""
        if ai_probability >= self.decision_threshold:
            return "AI_GENERATED", ai_probability
        return "HUMAN", 1.0 - ai_probability

    def decide(self, ai_probability, explanation_text, language):
        """Build the final API response dict with classification and explanation."""
        try:
            if not isinstance(ai_probability, (float, int)):
                raise DecisionOrchestrationError(
                    "Invalid ai_probability: must be a number"
                )
            if not 0.0 <= ai_probability <= 1.0:
                raise DecisionOrchestrationError(
                    "Invalid ai_probability: must be between 0.0 and 1.0"
                )

            if not isinstance(explanation_text, str) or not explanation_text.strip():
                explanation_text = "Explanation unavailable"

            classification, confidence_score = self.classify(ai_probability)

            return {
                "status": "success",
                "language": language,
                "classification": classification,
                "confidenceScore": round(confidence_score, 4),
                "explanation": explanation_text
            }

        except DecisionOrchestrationError:
            raise
        except Exception as e:
            raise DecisionOrchestrationError(
                f"Decision orchestration failed: {e}"
            ) from e
