"""Train the detectors and the fault classifier, and run them on new data.

    Isolation Forest   anomaly detection, trained on healthy windows only
    Random Forest      fault type, trained on labelled fault windows only
    RMS z-score        baseline detector (baseline.py), trained alongside

Thresholds are tuned on the validation split; the test split is never touched
here. Everything needed for inference is saved in one bundle (models/machinasense.joblib).

Usage:
    python model.py                      # CWRU
    python model.py --data my_data.csv   # hackathon CSV (add --window for slow sensors)
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import f1_score

from baseline import RMSZScoreDetector, pick_threshold
from data_loader import NORMAL_LABEL, load_dataset
from features import N_BANDS, STEP, WINDOW_SIZE, add_noise, drop_samples, extract_features, split_chronologically

SEED = 42
MODEL_PATH = Path(__file__).parent / "models" / "machinasense.joblib"

# Severity = how far the most deviating feature is from the healthy mean, in healthy std devs.
# Round decades: <10 sigma low, 10-100 sigma medium, >=100 sigma high.
SEVERITY_SIGMA = (10, 100)
TOP_K = 3

# Training-time augmentation: corrupted copies of the training split (0 = off).
AUGMENT_COPIES = 5
AUGMENT_SNR_DB = (5, 40)     # noise level per copy, dB relative to the healthy signal
AUGMENT_DROP = (0.0, 0.2)    # share of dropped samples drawn per copy


def _threshold(scores, labels, fallback_quantile=0.99):
    """F1-optimal threshold when validation has both classes, else a high quantile."""
    if labels is not None:
        abnormal = labels != NORMAL_LABEL
        if abnormal.any() and (~abnormal).any():
            return pick_threshold(scores, abnormal)
    return float(np.quantile(scores, fallback_quantile))


def augment(train_df, sensor_cols, reference_std, copies=AUGMENT_COPIES):
    """Append `copies` corrupted versions of the training data, so the models learn
    that a noisy sensor is not a faulty machine. Copy i draws its noise level from the
    i-th slice of AUGMENT_SNR_DB (so the whole range is covered) and a random drop rate."""
    rng = np.random.default_rng(SEED)
    frames, n_segments = [train_df], train_df.segment.max() + 1
    snr_edges = np.linspace(*AUGMENT_SNR_DB, copies + 1)
    for i in range(copies):
        snr, rate = rng.uniform(snr_edges[i], snr_edges[i + 1]), rng.uniform(*AUGMENT_DROP)
        noisy = add_noise(train_df, sensor_cols, snr, reference_std, seed=SEED + i)
        noisy = drop_samples(noisy, sensor_cols, rate, seed=SEED + 100 + i)
        noisy["segment"] += (i + 1) * n_segments          # keep segment ids unique
        frames.append(noisy)
    return pd.concat(frames, ignore_index=True)


def train(source="cwru", fs=None, window=WINDOW_SIZE, step=STEP, n_bands=N_BANDS, augment_copies=AUGMENT_COPIES):
    ds = load_dataset(source, fs=fs)
    splits = split_chronologically(ds.df)
    train_df, cols = splits["train"], ds.sensor_cols
    # Without labels we assume the training period is (mostly) healthy.
    has_normal = ds.has_labels and (train_df.label == NORMAL_LABEL).any()

    def featurize(df):
        X, windows = extract_features(df, ds.fs, cols, window, step, n_bands)
        healthy = (windows.label == NORMAL_LABEL).to_numpy() if has_normal else np.ones(len(X), bool)
        return X, windows, healthy

    X_train, w_train, healthy = featurize(train_df)
    X_val, w_val, _ = featurize(splits["val"])
    y_val = w_val["label"].to_numpy() if ds.has_labels else None
    healthy_rows = train_df.label == NORMAL_LABEL if has_normal else slice(None)
    signal_std = train_df.loc[healthy_rows, cols].std()   # reference level for noise

    # Baseline and the "healthy profile" (used for severity) come from clean healthy windows.
    baseline = RMSZScoreDetector().fit(X_train[healthy])
    baseline.threshold = _threshold(baseline.score(X_val), y_val)

    # The ML models also see corrupted copies of the training split.
    if augment_copies:
        X_aug, w_aug, healthy_aug = featurize(augment(train_df, cols, signal_std, augment_copies))
    else:
        X_aug, w_aug, healthy_aug = X_train, w_train, healthy
    iforest = IsolationForest(n_estimators=200, random_state=SEED).fit(X_aug[healthy_aug])
    if_threshold = _threshold(-iforest.score_samples(X_val), y_val)

    rf = None
    if ds.has_labels and (~healthy_aug).any():
        y_aug = w_aug["label"].to_numpy()
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=SEED, n_jobs=-1)
        rf.fit(X_aug[~healthy_aug], y_aug[~healthy_aug])

    bundle = {
        "data_source": ds.source, "fs": ds.fs, "sensor_cols": cols,
        "window": window, "step": step, "n_bands": n_bands, "has_labels": ds.has_labels,
        "augment_copies": augment_copies,
        "feature_names": list(X_train.columns),
        "healthy_mean": X_train[healthy].mean(),
        "healthy_std": X_train[healthy].std().clip(lower=1e-12),
        "signal_std": signal_std,
        "baseline": baseline, "iforest": iforest, "if_threshold": if_threshold, "rf": rf,
    }
    return bundle, X_val, w_val


def save_bundle(bundle, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path=MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found - run `python model.py` first")
    return joblib.load(path)


def deviation_sigma(features, bundle):
    """Signed distance of every feature from the healthy mean, in healthy std devs."""
    cols = bundle["feature_names"]
    return (features[cols] - bundle["healthy_mean"][cols]) / bundle["healthy_std"][cols]


def severity_of(max_sigma):
    low, high = SEVERITY_SIGMA
    return np.select([max_sigma >= high, max_sigma >= low], ["high", "medium"], "low")


def analyze(df, bundle):
    """Run the full pipeline on a sample-level DataFrame (data_loader format).

    Returns (results, features): one row per window. `results` has the window
    position/time, anomaly score, alert flag, severity, fault type, top features
    and the baseline's verdict for comparison.
    """
    features, out = extract_features(df, bundle["fs"], bundle["sensor_cols"],
                                      bundle["window"], bundle["step"], bundle["n_bands"])
    X = features[bundle["feature_names"]]

    out["anomaly_score"] = -bundle["iforest"].score_samples(X)
    out["is_alert"] = out.anomaly_score > bundle["if_threshold"]
    out["baseline_score"] = bundle["baseline"].score(X)
    out["baseline_alert"] = out.baseline_score > bundle["baseline"].threshold

    if bundle["rf"] is not None:
        proba = bundle["rf"].predict_proba(X)
        out["fault_type"] = bundle["rf"].classes_[proba.argmax(axis=1)]
        out["fault_confidence"] = proba.max(axis=1)
    else:
        out["fault_type"], out["fault_confidence"] = "unknown_anomaly", np.nan
    # System output: "normal" unless the anomaly detector fires, then the fault type.
    out["predicted"] = np.where(out.is_alert, out.fault_type, NORMAL_LABEL)
    out["baseline_predicted"] = np.where(out.baseline_alert, out.fault_type, NORMAL_LABEL)

    z = deviation_sigma(X, bundle)
    absz = z.abs().to_numpy()
    out["max_deviation_sigma"] = absz.max(axis=1)
    out["severity"] = np.where(out.is_alert, severity_of(out.max_deviation_sigma), "none")
    top = np.argsort(-absz, axis=1)[:, :TOP_K]
    names, zv = np.array(z.columns), z.to_numpy()
    out["top_features"] = [", ".join(f"{names[j]} ({zv[i, j]:+.0f}σ)" for j in row)
                           for i, row in enumerate(top)]
    return out, features


def explain(feature_row, bundle, k=8):
    """Table of the k features that deviate most from healthy for one window."""
    cols = bundle["feature_names"]
    value = feature_row[cols].astype(float)
    mean, std = bundle["healthy_mean"][cols], bundle["healthy_std"][cols]
    table = pd.DataFrame({"value": value, "healthy_mean": mean, "healthy_std": std,
                          "deviation_sigma": (value - mean) / std})
    table["x_healthy"] = value / mean.where(mean.abs() > 1e-9)
    return table.reindex(table.deviation_sigma.abs().sort_values(ascending=False).index).head(k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="cwru", help="'cwru' or a CSV path")
    parser.add_argument("--fs", type=float, default=None, help="sampling rate in Hz (CSV; inferred if omitted)")
    parser.add_argument("--window", type=int, default=WINDOW_SIZE, help="samples per window")
    parser.add_argument("--step", type=int, default=STEP, help="samples between window starts")
    parser.add_argument("--bands", type=int, default=N_BANDS, help="FFT band features per sensor (0 = none)")
    parser.add_argument("--no-augment", action="store_true", help="train on clean data only")
    args = parser.parse_args()

    copies = 0 if args.no_augment else AUGMENT_COPIES
    bundle, X_val, w_val = train(args.data, args.fs, args.window, args.step, args.bands, augment_copies=copies)
    save_bundle(bundle)
    print(f"trained on {bundle['data_source']}: {len(bundle['feature_names'])} features, "
          f"window={bundle['window']} step={bundle['step']} fs={bundle['fs']:g} Hz, "
          f"{copies} augmented copies")
    print(f"thresholds (tuned on validation): baseline |z| > {bundle['baseline'].threshold:.2f}, "
          f"isolation forest score > {bundle['if_threshold']:.4f}")
    if bundle["has_labels"]:
        abnormal = w_val.label.to_numpy() != NORMAL_LABEL
        for name, alerts in [("baseline", bundle["baseline"].predict(X_val)),
                             ("isolation forest", -bundle["iforest"].score_samples(X_val) > bundle["if_threshold"])]:
            print(f"validation detection F1 ({name}): {f1_score(abnormal, alerts, zero_division=0):.3f}")
    print(f"saved {MODEL_PATH}")
