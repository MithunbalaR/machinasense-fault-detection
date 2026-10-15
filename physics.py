"""Envelope analysis: does the vibration physics agree with the ML fault prediction?

A damaged bearing produces a burst of vibration every time a ball rolls over the
defect. Those bursts repeat at a frequency fixed by the bearing geometry and shaft
speed (inner race, outer race and ball each have their own). Band-pass filtering
the signal, taking its envelope and looking at the envelope's spectrum reveals
that repetition rate as a clear peak.

Needs the bearing's defect frequencies (Dataset.physics, set in data_loader.py).
"""
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from features import fill_missing

CONTEXT_SAMPLES = 16384      # samples around the alert window (~0.34 s at 48 kHz -> ~3 Hz resolution)
BAND = (0.04, 0.40)          # band-pass edges as a fraction of the sampling rate (resonance region)
TOLERANCE = 0.04             # search +-4% around each defect frequency (speed varies slightly)
HARMONICS = (1, 2)           # average the peak over the defect frequency and its 2nd harmonic
PEAK_DB = 12                 # a "clear" peak; healthy bearings sit around 5-6 dB


def available(physics, fs):
    """Envelope analysis only makes sense for fast vibration data with known defect frequencies."""
    return bool(physics) and fs >= 20 * max(physics["fault_hz"].values()) * max(HARMONICS)


def envelope_spectrum(x, fs):
    """Amplitude spectrum of the band-passed signal's envelope."""
    x = fill_missing(np.asarray(x, float)[:, None])[:, 0]
    sos = butter(4, [BAND[0] * fs, BAND[1] * fs], btype="band", fs=fs, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, x - x.mean())))
    env = (env - env.mean()) * np.hanning(len(env))
    return np.fft.rfftfreq(len(env), 1 / fs), np.abs(np.fft.rfft(env))


def peak_prominence(freqs, amp, fault_hz):
    """For each fault type: peak height near its defect frequency, in dB above the background."""
    top = 4 * max(fault_hz.values())
    background = np.median(amp[(freqs > 5) & (freqs < top)]) + 1e-12
    out = {}
    for label, f0 in fault_hz.items():
        peaks = [amp[(freqs > h * f0 * (1 - TOLERANCE)) & (freqs < h * f0 * (1 + TOLERANCE))].max() for h in HARMONICS]
        out[label] = float(np.mean(20 * np.log10(np.array(peaks) / background)))
    return out


def context(signal_df, start, window, n=CONTEXT_SAMPLES):
    """Positions of ~n samples centred on a window, staying inside its segment."""
    seg = signal_df.segment.to_numpy()
    same = np.flatnonzero(seg == seg[start])          # segments are contiguous runs of rows
    first, last = same[0], same[-1] + 1
    lo = max(first, start + window // 2 - n // 2)
    hi = min(last, lo + n)
    lo = max(first, hi - n)
    return slice(lo, hi)


def check(signal_df, start, window, fs, physics, predicted):
    """Physics verdict for one alert: 'confirms', 'disagrees' or 'inconclusive'."""
    x = signal_df[physics["channel"]].to_numpy()[context(signal_df, start, window)]
    freqs, amp = envelope_spectrum(x, fs)
    prominence = peak_prominence(freqs, amp, physics["fault_hz"])
    best = max(prominence, key=prominence.get)
    if prominence[best] < PEAK_DB:
        verdict = "inconclusive"
    elif best == predicted:
        verdict = "confirms"
    else:
        verdict = "disagrees"
    return {"verdict": verdict, "best": best, "prominence": prominence, "freqs": freqs, "amp": amp}
