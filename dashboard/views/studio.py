"""Data studio: bring your own CSV, analyse it with the active model or train a new one."""
import contextlib
import html
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import f1_score

import evaluate
from dashboard import state
from dashboard.charts import health_chart, score_chart
from dashboard.ui import card, hero, page_title, pretty, show, show_chart
from data_loader import NORMAL_LABEL, load_dataset
from maintenance import incidents
from model import AUGMENT_COPIES, save_bundle, train

NONE = "— none —"


@st.cache_resource(show_spinner="Loading dataset ...")
def load_prepared(path, mtime):
    return load_dataset(str(path))


def guess(columns, words):
    return next((c for c in columns if any(w in c.lower() for w in words)), None)


def prepare(raw, ts_col, label_col, sensors, path):
    """Write the upload in the standard CSV format (timestamp, sensors..., label)."""
    out = pd.DataFrame({"timestamp": raw[ts_col]})
    for s in sensors:
        out[s] = pd.to_numeric(raw[s], errors="coerce")
    if label_col != NONE:
        out["label"] = raw[label_col]
    out.to_csv(path, index=False)


def render():
    bundle = state.get_bundle()
    hero([f"active model: <b>{html.escape(str(bundle['data_source']))}</b>"])
    page_title("Data studio", "Bring your own data: upload a CSV, map its columns, then analyse it with the active "
                              "model or train a new model on it. This is the hackathon-day workflow, without code.")

    state.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    c1, c2 = st.columns([2, 1])
    upload = c1.file_uploader("1 · Upload a CSV with a timestamp column, sensor columns and (optionally) a label column",
                              type=["csv"])
    if c2.button("Use a sample file instead", help="About 4 s of the held-out CWRU test recordings (1 s per condition)", width="stretch"):
        sample = state.UPLOAD_DIR / "sample_cwru.csv"
        test = state.get_splits("cwru", 48000)["test"]
        test.groupby("segment").head(48000 * 3 // 2)[["timestamp", "DE", "FE", "label"]].to_csv(sample, index=False)
        st.session_state.studio_raw = str(sample)
    if upload is not None:
        raw_path = state.UPLOAD_DIR / upload.name
        raw_path.write_bytes(upload.getvalue())
        st.session_state.studio_raw = str(raw_path)
    raw_path = st.session_state.get("studio_raw")
    if not raw_path:
        st.info("Upload a file (or use the sample) to start.")
        return

    raw = pd.read_csv(raw_path)
    st.markdown(f"**2 · Map the columns** of `{html.escape(Path(raw_path).name)}` "
                f"({len(raw):,} rows, {len(raw.columns)} columns)")
    st.dataframe(raw.head(8), width="stretch", height=250)
    cols = list(raw.columns)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(raw[c])]
    m1, m2, m3 = st.columns([1, 1, 2])
    ts_col = m1.selectbox("Timestamp column", cols, index=cols.index(guess(cols, ["time", "date"]) or cols[0]))
    label_options = [NONE] + [c for c in cols if c != ts_col]
    label_guess = guess(cols, ["label", "fault", "class", "state", "status"])
    label_col = m2.selectbox("Label column (optional)", label_options,
                             index=label_options.index(label_guess) if label_guess in label_options else 0)
    sensors = m3.multiselect("Sensor columns", [c for c in numeric if c not in (ts_col, label_col)],
                             default=[c for c in numeric if c not in (ts_col, label_col)])
    if not sensors:
        st.warning("Pick at least one sensor column.")
        return
    prepared_path = state.UPLOAD_DIR / f"{Path(raw_path).stem}_prepared.csv"
    if st.button("Prepare dataset", type="primary"):
        prepare(raw, ts_col, label_col, sensors, prepared_path)
        st.session_state.studio_prepared = str(prepared_path)
    path = st.session_state.get("studio_prepared")
    if not path or not prepared_path.exists() or path != str(prepared_path):
        return

    # ------------------------------------------------------------------------- dataset summary
    ds = load_prepared(path, prepared_path.stat().st_mtime)
    st.markdown("**3 · Check the dataset**")
    k = st.columns(4)
    show(card("Samples", f"{len(ds.df):,}", f"{ds.df.timestamp.iloc[0]} →", f"{ds.df.timestamp.iloc[-1]}"), k[0])
    show(card("Sampling rate", f"{ds.fs:,.0f} Hz" if ds.fs >= 100 else f"{ds.fs:.3g} Hz", "inferred from the timestamps"), k[1])
    show(card("Sensors", len(ds.sensor_cols), ", ".join(map(str, ds.sensor_cols))[:80]), k[2])
    labels = ds.df.label.value_counts() if ds.has_labels else None
    show(card("Labels", len(labels) if labels is not None else "none",
              ", ".join(f"{pretty(i)} {v:,}" for i, v in labels.items())[:120] if labels is not None
              else "anomaly-only mode (no fault types)", small=True), k[3])

    tab_analyse, tab_train = st.tabs(["🔍 Analyse with the active model", "🏋 Train a new model on this data"])
    with tab_analyse:
        missing = [c for c in bundle["sensor_cols"] if c not in ds.sensor_cols]
        if missing or abs(ds.fs - bundle["fs"]) > 0.01 * bundle["fs"]:
            st.warning(f"The active model expects sensors {bundle['sensor_cols']} at {bundle['fs']:g} Hz. "
                       f"This file has {ds.sensor_cols} at {ds.fs:g} Hz. Train a new model on it instead.")
        elif st.button("Run analysis", type="primary"):
            st.session_state.studio_analysed = path
        if st.session_state.get("studio_analysed") == path and not missing:
            results, _, health = state.run_dataframe(ds.df)
            inc = incidents(results)
            a = st.columns(4)
            show(card("Windows", f"{len(results):,}", f"{bundle['window']} samples each"), a[0])
            show(card("Alerts", int(results.is_alert.sum()), f"{results.is_alert.mean():.0%} of windows"), a[1])
            show(card("Incidents", len(inc), ", ".join(sorted({pretty(f) for f in inc.fault_type}))[:80] or "none", small=True), a[2])
            if "label" in results:
                truth = (results.label != NORMAL_LABEL).to_numpy()
                show(card("Detection F1 vs labels", f"{f1_score(truth, results.is_alert, zero_division=0):.2f}",
                          "uses the file's labels; this file was not used for training"), a[3])
            show_chart(score_chart(results, bundle["if_threshold"], 260),
                       f"Anomaly score for {len(results)} windows of the uploaded data; {int(results.is_alert.sum())} are "
                       f"above the alert threshold {bundle['if_threshold']:.3f}.")
            show_chart(health_chart(health, bundle["step"] / bundle["fs"]),
                       f"Health index of the uploaded data: lowest {health.min():.0f}, final {health[-1]:.0f} of 100.")
            st.dataframe(inc.assign(fault=inc.fault_type.map(pretty)).drop(columns=["fault_type"]), width="stretch")

    with tab_train:
        fast = ds.fs >= 1000
        st.markdown("Trains the full pipeline (baseline, Isolation Forest, Random Forest) with a chronological "
                    "60/20/20 split per recording, and makes it the **active model** for every page.")
        t = st.columns(4)
        window = t[0].number_input("Window (samples)", 8, 65536, 2048 if fast else 60, help="Span the time a fault takes to show")
        step = t[1].number_input("Step (samples)", 1, 65536, 1024 if fast else 30)
        bands = t[2].number_input("FFT bands per sensor", 0, 32, 8 if fast else 0, help="0 for slow (SCADA-like) data")
        copies = t[3].number_input("Augmented copies", 0, 10, AUGMENT_COPIES, help="Noise/drop augmentation for robustness")
        st.caption("This replaces models/machinasense.joblib. Restore the CWRU demo model any time with the button below "
                   "or `python model.py`.")
        b1, b2, b3 = st.columns(3)
        if b1.button("Train & activate", type="primary", width="stretch"):
            with st.spinner("Training ..."):
                new_bundle, _, _ = train(path, None, int(window), int(step), int(bands), augment_copies=int(copies))
                save_bundle(new_bundle)
            state.reset_caches()
            st.success(f"Trained on {len(new_bundle['feature_names'])} features; alert threshold "
                       f"{new_bundle['if_threshold']:.3f}. All pages now use this model.")
        if b2.button("Run evaluation on the active model", width="stretch"):
            buffer = io.StringIO()
            with st.spinner("Evaluating (held-out split, robustness, physics, generalisation) ..."), \
                    contextlib.redirect_stdout(buffer):
                evaluate.main()
            st.code(buffer.getvalue()[-6000:] or "done", language="text")
        if b3.button("Restore the CWRU demo model", width="stretch"):
            with st.spinner("Training on CWRU ..."):
                save_bundle(train("cwru")[0])
            state.reset_caches()
            st.success("CWRU demo model restored. Run the evaluation to refresh the Model lab.")
