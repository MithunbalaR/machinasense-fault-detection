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
# Optional bearing physics for the envelope check (physics.py), e.g. for a vibration CSV:
# CSV_PHYSICS = {"channel": "vibration", "shaft_rpm": 1800,
#                "orders": {"inner_race": 5.4152, "outer_race": 3.5848, "ball": 4.7135}}
CSV_PHYSICS = None

# --- CWRU subset: 48 kHz drive-end experiments, 0.007" faults, motor loads 0-3 HP ---
CWRU_URL = "https://engineering.case.edu/sites/default/files/{}.mat"
CWRU_FS = 48_000
CWRU_LOADS = {               # motor load (HP) -> file number per condition
    0: {"normal": 97, "inner_race": 109, "ball": 122, "outer_race": 135},
    1: {"normal": 98, "inner_race": 110, "ball": 123, "outer_race": 136},
    2: {"normal": 99, "inner_race": 111, "ball": 124, "outer_race": 137},
    3: {"normal": 100, "inner_race": 112, "ball": 125, "outer_race": 138},
}
CWRU_FILES = CWRU_LOADS[0]       # the load used for training
CWRU_CHANNELS = ["DE", "FE"]     # drive-end and fan-end accelerometers
CWRU_START = pd.Timestamp("2026-01-05 08:00:00")  # CWRU has no clock; timestamps are synthetic
# Drive-end bearing SKF 6205: defect frequencies as multiples of shaft speed (CWRU website).
CWRU_BEARING_ORDERS = {"inner_race": 5.4152, "outer_race": 3.5848, "ball": 4.7135}


@dataclass
class Dataset:
    df: pd.DataFrame
    fs: float
    sensor_cols: list
    has_labels: bool
    source: str
    physics: dict = None     # {"channel", "shaft_hz", "fault_hz": {label: Hz}} if bearing geometry is known

    def summary(self):
        lines = [f"source: {self.source}", f"samples: {len(self.df):,}  fs: {self.fs:g} Hz",
                 f"sensors: {self.sensor_cols}",
                 f"time: {self.df.timestamp.iloc[0]} -> {self.df.timestamp.iloc[-1]}",
                 f"segments: {self.df.segment.nunique()}"]
        if self.has_labels:
            counts = self.df.groupby("label").size()
            lines.append("samples per label: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
        if self.physics:
            lines.append("bearing defect frequencies: " + ", ".join(
                f"{k}={v:.1f} Hz" for k, v in self.physics["fault_hz"].items()))
        return "\n".join(lines)


def make_physics(channel, shaft_rpm, orders):
    shaft_hz = shaft_rpm / 60
    return {"channel": channel, "shaft_hz": shaft_hz, "fault_hz": {k: o * shaft_hz for k, o in orders.items()}}


def load_dataset(source="cwru", fs=None):
    """Load any supported source into the standard Dataset format."""
    if source == "cwru":
        return load_cwru()
    if str(source).lower().endswith(".csv"):
        return load_csv(source, fs=fs)
    raise ValueError(f"Unknown data source {source!r}: use 'cwru' or a .csv path")


# ----------------------------------------------------------------------------- CWRU
def download_cwru(raw_dir=RAW_DIR, files=CWRU_FILES):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for label, num in files.items():
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


def load_cwru(load_hp=0, raw_dir=RAW_DIR):
    files = CWRU_LOADS[load_hp]
    download_cwru(raw_dir, files)
    frames, rpms, start = [], [], CWRU_START
    for segment, (label, num) in enumerate(files.items()):
        mat = scipy.io.loadmat(raw_dir / f"{num}.mat")
        rpms += [float(mat[k].ravel()[0]) for k in mat if k.endswith("RPM")]
        signals = {ch: _read_cwru_channel(mat, num, ch) for ch in CWRU_CHANNELS}
        n = min(len(x) for x in signals.values())
        frame = pd.DataFrame({ch: x[:n] for ch, x in signals.items()})
        frame.insert(0, "timestamp", start + pd.to_timedelta(np.arange(n) / CWRU_FS, unit="s"))
        frame["label"] = label
        frame["segment"] = segment
        frames.append(frame)
        start = frame.timestamp.iloc[-1] + pd.Timedelta(seconds=10)  # recordings back to back
    df = pd.concat(frames, ignore_index=True)
    physics = make_physics("DE", float(np.median(rpms)), CWRU_BEARING_ORDERS)
    source = "cwru" if load_hp == 0 else f"cwru ({load_hp} HP)"
    return Dataset(df, CWRU_FS, list(CWRU_CHANNELS), True, source, physics)


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
    physics = None
    if CSV_PHYSICS:
        physics = make_physics(CSV_PHYSICS["channel"], CSV_PHYSICS["shaft_rpm"], CSV_PHYSICS["orders"])
    return Dataset(df[keep], float(fs), sensor_cols, bool(has_labels), str(path), physics)


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
