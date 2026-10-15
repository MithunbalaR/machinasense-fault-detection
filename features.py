"""Chronological splitting, windowing and per-window features.

Features per sensor channel and window:
    mean, rms, peak (max |x|), crest factor (peak / rms), kurtosis (3 = Gaussian),
    skewness, and the share of spectral energy in N_BANDS equal-width FFT bands.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import kurtosis, skew

WINDOW_SIZE = 2048          # samples per window (~43 ms at 48 kHz, ~1.3 shaft revolutions)
STEP = 1024                 # hop between windows (50% overlap)
N_BANDS = 8                 # FFT bands between 0 Hz and Nyquist (0 = no spectral features)
SPLIT = (0.6, 0.2, 0.2)     # train / validation / test share of every segment


def split_chronologically(df, fractions=SPLIT):
    """Cut every segment into train / val / test by time (earliest -> latest).

    Splitting each recording by time (instead of shuffling windows) keeps
    overlapping windows from leaking between train and test, and every
    class still appears in every split.
    """
    parts = {"train": [], "val": [], "test": []}
    for _, seg in df.groupby("segment", sort=False):
        n = len(seg)
        a, b = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
        parts["train"].append(seg.iloc[:a])
        parts["val"].append(seg.iloc[a:b])
        parts["test"].append(seg.iloc[b:])
    return {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}


def fill_missing(x):
    """Linearly interpolate NaNs (dropped samples) in each column of a 2-D array."""
    x = x.copy()
    idx = np.arange(len(x))
    for c in range(x.shape[1]):
        bad = np.isnan(x[:, c])
        if bad.all():
            x[:, c] = 0.0
        elif bad.any():
            x[bad, c] = np.interp(idx[bad], idx[~bad], x[~bad, c])
    return x


def band_edges(fs, n_bands):
    return np.linspace(0, fs / 2, n_bands + 1)


def _band_name(lo, hi):
    if hi >= 1000:
        return f"band_{lo / 1000:.3g}-{hi / 1000:.3g}kHz"
    return f"band_{lo:.3g}-{hi:.3g}Hz"


def feature_names(sensor_cols, fs, n_bands=N_BANDS):
    edges = band_edges(fs, n_bands)
    per_channel = ["mean", "rms", "peak", "crest", "kurtosis", "skewness"]
    per_channel += [_band_name(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
    return [f"{ch}_{name}" for ch in sensor_cols for name in per_channel]


def _channel_features(w, fs, n_bands):
    """w: (n_windows, window_size) for one channel -> (n_windows, n_features)."""
    rms = np.sqrt(np.mean(w ** 2, axis=1))
    peak = np.max(np.abs(w), axis=1)
    crest = peak / np.maximum(rms, 1e-12)
    kurt = kurtosis(w, axis=1, fisher=False)
    skw = skew(w, axis=1)
    cols = [w.mean(axis=1), rms, peak, crest, kurt, skw]
    if n_bands == 0:
        return np.nan_to_num(np.column_stack(cols))

    centered = w - w.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    freqs = np.fft.rfftfreq(w.shape[1], d=1 / fs)
    edges = band_edges(fs, n_bands)
    bands = np.stack([power[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
                      for lo, hi in zip(edges[:-1], edges[1:])], axis=1)
    bands[:, -1] += power[:, freqs == edges[-1]].sum(axis=1)     # include the Nyquist bin
    band_share = bands / np.maximum(bands.sum(axis=1, keepdims=True), 1e-12)
    return np.nan_to_num(np.column_stack(cols + [band_share]))


def extract_features(df, fs, sensor_cols, window=WINDOW_SIZE, step=STEP, n_bands=N_BANDS):
    """Slide windows over every segment of df and compute features.

    Returns (features, windows):
        features  DataFrame, one row per window, one column per feature
        windows   DataFrame, one row per window: segment, start (row position
                  in df), timestamp (window start), end_time, label (if any)
    """
    feats, meta = [], []
    values = df[sensor_cols].to_numpy(float)
    times = df.timestamp.to_numpy()
    for seg_id, seg_pos in df.groupby("segment", sort=False).indices.items():
        if len(seg_pos) < window:
            continue
        x = fill_missing(values[seg_pos])
        starts = np.arange(0, len(seg_pos) - window + 1, step)
        per_channel = [_channel_features(sliding_window_view(x[:, c], window)[starts], fs, n_bands)
                       for c in range(len(sensor_cols))]
        feats.append(np.hstack(per_channel))

        rows = seg_pos[starts]
        info = pd.DataFrame({
            "segment": seg_id,
            "start": rows,
            "timestamp": times[rows],
            "end_time": times[rows + window - 1],
        })
        if "label" in df:
            # Segments hold a single label, so the label at the window start is the window's label.
            info["label"] = df.label.to_numpy()[rows]
        meta.append(info)

    if not feats:
        raise ValueError(f"No segment is long enough for a {window}-sample window")
    features = pd.DataFrame(np.vstack(feats), columns=feature_names(sensor_cols, fs, n_bands))
    return features, pd.concat(meta, ignore_index=True)


# ----------------------------------------------------------------------------- corruptions
# Used for training-time augmentation (model.py) and the robustness test (evaluate.py).
def add_noise(df, sensor_cols, snr_db, reference_std, seed=0):
    """Add white Gaussian noise. SNR is relative to the healthy signal's std, so the
    noise floor is the same for every machine state (like a noisy sensor)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for col in sensor_cols:
        noise_std = reference_std[col] / 10 ** (snr_db / 20)
        out[col] = out[col] + rng.normal(0, noise_std, len(out))
    return out


def drop_samples(df, sensor_cols, rate, seed=0, fill="nan"):
    """Randomly lose a fraction of readings. 'nan' leaves gaps for fill_missing to
    interpolate (what this pipeline does); 'zero' mimics a naive logger."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    lost = rng.random(len(out)) < rate
    out.loc[lost, sensor_cols] = np.nan if fill == "nan" else 0.0
    return out
