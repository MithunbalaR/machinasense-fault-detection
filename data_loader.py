"""Data loading - the ONLY module that knows about raw data formats.

Everything downstream works with a `Dataset`, whose `df` has one row per sample:

    timestamp   datetime64
    <sensors>   one float column per sensor channel (NaN = missing reading)
    label       condition label, e.g. "normal", "inner_race" (only if labels exist)
    segment     int id of a contiguous recording; windows never cross segments

Supported sources:
    "cwru"            CWRU bearing subset (downloaded to data/raw on first use)
    "path/to/x.csv"   CSV with columns: timestamp, <sensor columns...>, [label]

Usage:
    python data_loader.py                          # download CWRU + print summary
    python data_loader.py --source my_data.csv     # check that a CSV loads correctly
    python data_loader.py --export-csv out.csv     # write CWRU in the CSV format
"""
import argparse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

RAW_DIR = Path(__file__).parent / "data" / "raw"
NORMAL_LABEL = "normal"

# --- CSV format (edit these on hackathon day if their column names differ) ---
CSV_TIMESTAMP_COL = "timestamp"
CSV_LABEL_COL = "label"
CSV_EXCLUDE_COLS = []            # numeric columns that are NOT sensors, e.g. ["machine_id"]
NORMAL_ALIASES = {"normal", "healthy", "ok", "good", "none", "0", "no_fault"}
GAP_FACTOR = 5                   # a time gap > 5x the usual spacing starts a new segment

# --- CWRU subset: 48 kHz drive-end experiments, 0 HP load (~1797 rpm), 0.007" faults ---
CWRU_URL = "https://engineering.case.edu/sites/default/files/{}.mat"
CWRU_FS = 48_000
CWRU_FILES = {"normal": 97, "inner_race": 109, "ball": 122, "outer_race": 135}
CWRU_CHANNELS = ["DE", "FE"]     # drive-end and fan-end accelerometers
CWRU_START = pd.Timestamp("2026-01-05 08:00:00")  # CWRU has no clock; timestamps are synthetic


@dataclass
class Dataset:
    df: pd.DataFrame
    fs: float
    sensor_cols: list
    has_labels: bool
    source: str

    def summary(self):
        lines = [f"source: {self.source}", f"samples: {len(self.df):,}  fs: {self.fs:g} Hz",
                 f"sensors: {self.sensor_cols}",
                 f"time: {self.df.timestamp.iloc[0]} -> {self.df.timestamp.iloc[-1]}",
                 f"segments: {self.df.segment.nunique()}"]
        if self.has_labels:
            counts = self.df.groupby("label").size()
            lines.append("samples per label: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
        return "\n".join(lines)


def load_dataset(source="cwru", fs=None):
    """Load any supported source into the standard Dataset format."""
    if source == "cwru":
        return load_cwru()
    if str(source).lower().endswith(".csv"):
        return load_csv(source, fs=fs)
    raise ValueError(f"Unknown data source {source!r}: use 'cwru' or a .csv path")


# ----------------------------------------------------------------------------- CWRU
def download_cwru(raw_dir=RAW_DIR):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for label, num in CWRU_FILES.items():
        path = raw_dir / f"{num}.mat"
        if not path.exists():
            print(f"downloading {label} ({num}.mat) ...")
            urllib.request.urlretrieve(CWRU_URL.format(num), path)


def _read_cwru_channel(mat, num, channel):
    keys = [k for k in mat if k.endswith(f"_{channel}_time")]
    # A few CWRU files contain extra variables; prefer the one named after the file.
    keys.sort(key=lambda k: f"{num:03d}" not in k)
    if not keys:
        raise KeyError(f"{num}.mat has no {channel} channel")
    return mat[keys[0]].ravel().astype(float)


def load_cwru(raw_dir=RAW_DIR):
    download_cwru(raw_dir)
    frames, start = [], CWRU_START
    for segment, (label, num) in enumerate(CWRU_FILES.items()):
        mat = scipy.io.loadmat(raw_dir / f"{num}.mat")
        signals = {ch: _read_cwru_channel(mat, num, ch) for ch in CWRU_CHANNELS}
        n = min(len(x) for x in signals.values())
        frame = pd.DataFrame({ch: x[:n] for ch, x in signals.items()})
        frame.insert(0, "timestamp", start + pd.to_timedelta(np.arange(n) / CWRU_FS, unit="s"))
        frame["label"] = label
        frame["segment"] = segment
        frames.append(frame)
        start = frame.timestamp.iloc[-1] + pd.Timedelta(seconds=10)  # recordings back to back
    df = pd.concat(frames, ignore_index=True)
    return Dataset(df, CWRU_FS, list(CWRU_CHANNELS), True, "cwru")


# ----------------------------------------------------------------------------- CSV
def load_csv(path, fs=None):
    df = pd.read_csv(path)
    if CSV_TIMESTAMP_COL not in df:
        raise ValueError(f"{path} needs a '{CSV_TIMESTAMP_COL}' column; found {list(df.columns)}")

    ts = df[CSV_TIMESTAMP_COL]
    # Numeric timestamps are treated as seconds (epoch or since start).
    df["timestamp"] = pd.to_datetime(ts, unit="s") if pd.api.types.is_numeric_dtype(ts) else pd.to_datetime(ts)
    if CSV_TIMESTAMP_COL != "timestamp":
        df = df.drop(columns=CSV_TIMESTAMP_COL)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    has_labels = CSV_LABEL_COL in df and df[CSV_LABEL_COL].notna().any()
    if has_labels:
        labels = df.pop(CSV_LABEL_COL).ffill().bfill().astype(str).str.strip().str.lower()
        df["label"] = labels.where(~labels.isin(NORMAL_ALIASES), NORMAL_LABEL)

    reserved = {"timestamp", "label", "segment", *CSV_EXCLUDE_COLS}
    sensor_cols = []
    for col in [c for c in df.columns if c not in reserved]:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().mean() > 0.5:          # mostly numeric -> a sensor
            df[col] = values
            sensor_cols.append(col)
    if not sensor_cols:
        raise ValueError(f"No numeric sensor columns found in {path}")

    dt = df.timestamp.diff().dt.total_seconds()
    median_dt = dt.median()
    if fs is None:
        fs = 1.0 / median_dt

    # New segment whenever the label changes or there is a large time gap.
    new_segment = dt.isna() | (dt > GAP_FACTOR * median_dt)
    if has_labels:
        new_segment |= df.label != df.label.shift()
    df["segment"] = new_segment.cumsum() - 1

    keep = ["timestamp", *sensor_cols] + (["label"] if has_labels else []) + ["segment"]
    return Dataset(df[keep], float(fs), sensor_cols, bool(has_labels), str(path))


def export_csv(dataset, path):
    """Write a dataset in the CSV format above (handy to test the CSV path)."""
    cols = ["timestamp", *dataset.sensor_cols] + (["label"] if dataset.has_labels else [])
    dataset.df[cols].to_csv(path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="cwru", help="'cwru' or a CSV path")
    parser.add_argument("--fs", type=float, default=None, help="sampling rate (CSV only; inferred if omitted)")
    parser.add_argument("--export-csv", help="write the loaded data to this CSV path")
    args = parser.parse_args()

    ds = load_dataset(args.source, fs=args.fs)
    print(ds.summary())
    if args.export_csv:
        export_csv(ds, args.export_csv)
        print(f"wrote {args.export_csv}")
