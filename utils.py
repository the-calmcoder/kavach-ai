"""Shared utility functions for waveform processing."""


def ensure_1d_numpy(waveform):
    """Convert waveform to a 1D numpy array regardless of input format."""
    if hasattr(waveform, 'numpy'):
        waveform = waveform.numpy()
    if len(waveform.shape) > 1:
        waveform = waveform.squeeze()
    return waveform
