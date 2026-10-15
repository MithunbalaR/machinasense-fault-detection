"""Baseline detector: z-score of window RMS against the healthy baseline.

A window is abnormal when, on any channel, its RMS is more than `threshold`
standard deviations away from the mean RMS of healthy training windows.
"""
import numpy as np
from sklearn.metrics import f1_score


class RMSZScoreDetector:
    def __init__(self):
        self.rms_cols, self.mean, self.std, self.threshold = None, None, None, None

    def fit(self, healthy_features):
        self.rms_cols = [c for c in healthy_features.columns if c.endswith("_rms")]
        self.mean = healthy_features[self.rms_cols].mean().to_numpy()
        self.std = np.maximum(healthy_features[self.rms_cols].std().to_numpy(), 1e-12)
        return self

    def score(self, features):
        """Largest |z| of RMS over all channels (higher = more abnormal)."""
        z = (features[self.rms_cols].to_numpy() - self.mean) / self.std
        return np.abs(z).max(axis=1)

    def predict(self, features):
        return self.score(features) > self.threshold


def pick_threshold(scores, is_abnormal):
    """Alert threshold that maximises F1 on validation data.

    Many thresholds often tie for the best F1 (the classes are separable), so
    take the middle of that range: it leaves the most margin on both sides.
    """
    values = np.unique(scores)
    candidates = np.concatenate([[values[0] - 1e-9], (values[:-1] + values[1:]) / 2])
    f1 = np.array([f1_score(is_abnormal, scores > t, zero_division=0) for t in candidates])
    best = candidates[np.isclose(f1, f1.max())]
    return float((best.min() + best.max()) / 2)
