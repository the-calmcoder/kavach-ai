"""Core inference engine using HuBERT for AI voice detection."""

import logging
import os

import numpy as np
import torch
from torch import nn
from transformers import Wav2Vec2FeatureExtractor, HubertModel

from utils import ensure_1d_numpy

logger = logging.getLogger(__name__)

HUBERT_MODEL_NAME = "facebook/hubert-base-ls960"
HUBERT_HIDDEN_SIZE = 768
DEFAULT_WEIGHTS_PATH = "classifier_weights.pth"
FALLBACK_PROBABILITY = 0.5


class AudioInferenceEngine:
    """Runs inference on audio waveforms using a fine-tuned HuBERT model."""

    def __init__(
        self,
        model_name=HUBERT_MODEL_NAME,
        weights_path=DEFAULT_WEIGHTS_PATH
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("Initializing Inference Engine on %s", self.device)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        self.model = HubertModel.from_pretrained(
            model_name
        ).to(self.device)
        self.model.eval()

        self.classifier = nn.Linear(HUBERT_HIDDEN_SIZE, 1).to(self.device)

        if os.path.exists(weights_path):
            logger.info("Loading trained weights from %s", weights_path)
            self.classifier.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
        else:
            logger.warning(
                "CRITICAL: %s not found! Model will predict randomly.",
                weights_path
            )

        self.classifier.eval()

    def infer(self, waveform, sample_rate):
        """Run inference and return (ai_probability, embedding)."""
        try:
            waveform = ensure_1d_numpy(waveform)

            inputs = self.feature_extractor(
                waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
            inputs = {
                k: v.to(self.device) for k, v in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
                logits = self.classifier(embedding)
                ai_probability = torch.sigmoid(logits).item()

            return ai_probability, embedding.cpu().numpy()

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                "Inference failed: %s — returning fallback", e
            )
            return FALLBACK_PROBABILITY, np.zeros((1, HUBERT_HIDDEN_SIZE))
