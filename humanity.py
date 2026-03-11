"""Builds a human voice baseline profile from audio samples."""

import json
import logging
import os
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac")
DEFAULT_OUTPUT_FILE = "human_baseline.json"


def get_features(file_path):
    """Extract ZCR, spectral flatness, and spectral centroid from an audio file."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        centroid = np.mean(
            librosa.feature.spectral_centroid(y=y, sr=sr)
        )
        return [float(zcr), float(flatness), float(centroid)]
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "Failed to process %s: %s",
            os.path.basename(file_path), e
        )
        return None


def build_baseline(data_path, output_path=DEFAULT_OUTPUT_FILE):
    """Compute mean/std for audio features and save as JSON."""
    if not os.path.exists(data_path):
        logger.error("Data folder '%s' does not exist.", data_path)
        return None

    files = [
        f for f in os.listdir(data_path)
        if f.endswith(AUDIO_EXTENSIONS)
    ]
    logger.info("Found %d audio files in %s", len(files), data_path)

    features_list = []
    for f in files:
        feats = get_features(os.path.join(data_path, f))
        if feats:
            features_list.append(feats)

    if not features_list:
        logger.error("No valid features extracted from any file.")
        return None

    data_matrix = np.array(features_list)
    means = np.mean(data_matrix, axis=0)
    stds = np.std(data_matrix, axis=0)

    profile = {
        "zcr": {"mean": float(means[0]), "std": float(stds[0])},
        "flatness": {"mean": float(means[1]), "std": float(stds[1])},
        "centroid": {"mean": float(means[2]), "std": float(stds[2])}
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f)

    logger.info("Baseline saved to %s", output_path)
    return profile


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    baseline_data_path = Path(__file__).parent / "data" / "human"
    result = build_baseline(str(baseline_data_path))
    if result:
        print(json.dumps(result, indent=2))
