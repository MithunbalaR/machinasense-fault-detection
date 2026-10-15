"""Data the pages share: the active model, the demo fleet, simulation settings and work orders."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from data_loader import NORMAL_LABEL, load_dataset
from dashboard.ui import REPORT_PATH, pretty
from features import add_noise, drop_samples, split_chronologically
from maintenance import health_index, recommend
from model import SEED, analyze, load_bundle

ROOT = Path(__file__).parent.parent
WORK_ORDERS_PATH = ROOT / "data" / "work_orders.json"
UPLOAD_DIR = ROOT / "data" / "uploads"
DUE_IN = {"high": timedelta(hours=24), "medium": timedelta(days=7), "low": timedelta(days=30)}
WORK_ORDER_STATUSES = ["Open", "In progress", "Done"]


# ----------------------------------------------------------------------------- model + data
@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_resource
def get_splits(source, fs):
    ds = load_dataset(source, fs=fs)
    return split_chronologically(ds.df) | {"all": ds.df}


def reset_caches():
    """Call after the active model changes (e.g. trained in the Data studio)."""
    st.cache_resource.clear()
    st.cache_data.clear()


def healthy_reference(bundle):
    """First healthy window of the training split, to compare alerts against."""
    train = get_splits(bundle["data_source"], bundle["fs"])["train"]
    if "label" in train and (train.label == NORMAL_LABEL).any():
        train = train[train.label == NORMAL_LABEL]
    return train.iloc[: bundle["window"]]


def load_report():
    return json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else None


# ----------------------------------------------------------------------------- simulation (sidebar)
def simulation():
    """Current sensor-problem simulation chosen in the sidebar: (snr_db or 'off', drop rate)."""
    return st.session_state.get("sim_snr", "off"), st.session_state.get("sim_drop", 0.0)


def corrupt(df, bundle, snr_db, drop_rate):
    cols = bundle["sensor_cols"]
    if snr_db != "off":
        df = add_noise(df, cols, snr_db, bundle["signal_std"], seed=SEED)
    if drop_rate > 0:
        df = drop_samples(df, cols, drop_rate, seed=SEED)
    return df


# ----------------------------------------------------------------------------- demo fleet
@st.cache_data(show_spinner=False)
def fleet_specs(source, fs):
    """Demo machines built from the held-out test split (never used for training or tuning).

    With labels: one machine that stays healthy, and one per fault type that runs
    healthy and then develops that fault. Without labels: one machine, the whole test split.
    """
    test = get_splits(source, fs)["test"]
    if "label" not in test:
        return {"M-101": {"name": "Machine M-101", "area": "Line A", "segments": [int(s) for s in test.segment.unique()],
                          "scenario": "Held-out test data"}}
    seg_of = {lab: int(test.segment[test.label == lab].iloc[0]) for lab in test.label.unique()}
    faults = sorted(lab for lab in seg_of if lab != NORMAL_LABEL)
    normal = [seg_of[NORMAL_LABEL]] if NORMAL_LABEL in seg_of else []
    specs = {}
    if normal:
        specs["M-101"] = {"name": "Motor M-101", "area": "Line A", "segments": normal,
                          "scenario": "Healthy test recording only"}
    for i, fault in enumerate(faults):
        mid = f"M-{102 + i}"
        specs[mid] = {"name": f"Motor {mid}", "area": "Line A" if i % 2 else "Line B", "segments": normal + [seg_of[fault]],
                      "scenario": f"Healthy test recording, then the {pretty(fault).lower()} test recording"}
    return specs


@st.cache_data(show_spinner=False)
def machine_signal(source, fs, machine_id):
    """Sample-level stream for one demo machine: its test segments, re-timed to run back to back."""
    test = get_splits(source, fs)["test"]
    spec = fleet_specs(source, fs)[machine_id]
    parts, start = [], pd.Timestamp("2026-01-05 08:00:00")
    for seg in spec["segments"]:
        part = test[test.segment == seg].copy()
        dt = part.timestamp.diff().median() if len(part) > 1 else pd.Timedelta(seconds=1)
        part["timestamp"] = start + (part.timestamp - part.timestamp.iloc[0])
        start = part.timestamp.iloc[-1] + dt
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


@st.cache_data(show_spinner="Analysing machine data ...")
def run_machine(machine_id, snr_db, drop_rate):
    """(signal_df, results, features, health) for one machine under the current simulation."""
    bundle = get_bundle()
    df = corrupt(machine_signal(bundle["data_source"], bundle["fs"], machine_id), bundle, snr_db, drop_rate)
    results, features = analyze(df, bundle)
    return df, results, features, health_index(results.severity)


@st.cache_data(show_spinner="Analysing uploaded data ...")
def run_dataframe(df):
    bundle = get_bundle()
    results, features = analyze(df, bundle)
    return results, features, health_index(results.severity)


def selected_machine(machines):
    mid = st.session_state.get("machine")
    return mid if mid in machines else next(iter(machines))


# ----------------------------------------------------------------------------- work orders
def load_orders():
    if WORK_ORDERS_PATH.exists():
        return json.loads(WORK_ORDERS_PATH.read_text(encoding="utf-8"))
    return []


def save_orders(orders):
    WORK_ORDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORK_ORDERS_PATH.write_text(json.dumps(orders, indent=2), encoding="utf-8")


def create_work_order(machine_id, machine_name, incident):
    """Create a work order for an incident unless one already exists. Returns (order, created?)."""
    orders = load_orders()
    key = f"{machine_id}|{incident['start_time']}|{incident['fault_type']}"
    existing = next((o for o in orders if o.get("key") == key), None)
    if existing:
        return existing, False
    now = datetime.now()
    rec = recommend(incident["fault_type"], incident["severity"])
    order = {
        "id": f"WO-{len(orders) + 1:04d}", "key": key, "machine": machine_id, "machine_name": machine_name,
        "fault": pretty(incident["fault_type"]), "severity": incident["severity"],
        "detected": str(incident["start_time"])[:23], "windows": int(incident["windows"]),
        "action": rec["action"], "urgency": rec["urgency"],
        "created": now.strftime("%Y-%m-%d %H:%M"), "due": (now + DUE_IN.get(incident["severity"], timedelta(days=7))).strftime("%Y-%m-%d %H:%M"),
        "status": "Open", "assignee": "", "notes": "",
    }
    orders.append(order)
    save_orders(orders)
    return order, True



@st.cache_data(show_spinner="Scoring the split ...")
def run_split(split):
    """Clean results for a whole split of the active model's data ('train', 'val' or 'test')."""
    bundle = get_bundle()
    results, _ = analyze(get_splits(bundle["data_source"], bundle["fs"])[split], bundle)
    return results
