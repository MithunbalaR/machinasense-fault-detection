"""MachinaSense dashboard: machine health, live replay, alerts with evidence, evaluation.

Run:  streamlit run app.py
      (after `python model.py`; `python evaluate.py` fills the Evaluation tab)
"""
import html
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from data_loader import NORMAL_LABEL, load_dataset
from features import add_noise, drop_samples, fill_missing, split_chronologically
from model import SEED, SEVERITY_SIGMA, analyze, explain, load_bundle

REPORT_PATH = Path(__file__).parent / "reports" / "metrics.json"
SEVERITIES = ["high", "medium", "low"]
SEVERITY_WEIGHT = {"none": 0.0, "low": 1 / 3, "medium": 2 / 3, "high": 1.0}
RECENT = 20                      # windows behind the "machine health" score
REPLAY_SPEEDS = {"0.5×": 0.8, "1×": 0.4, "2×": 0.2, "4×": 0.1}   # seconds per window

# Colours: dark-surface steps of a validated palette. Status colours always come with a text label.
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEVERITY_COLORS = {"none": "#898781", "low": STATUS["warning"], "medium": STATUS["serious"], "high": STATUS["critical"]}
SEVERITY_ICONS = {"high": "🔴", "medium": "🟠", "low": "🟡", "none": "⚪"}
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
INK, INK_2, MUTED, SURFACE = "#f0efec", "#c3c2b7", "#898781", "#1a1a19"

CSS = """
[data-testid="stMainBlockContainer"] { padding-top: 3.6rem; max-width: 1440px; }
.ms-hero { display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 16px;
  padding: 22px 26px; margin-bottom: 18px; border-radius: 16px; border: 1px solid rgba(255,255,255,.08);
  background: radial-gradient(1200px 300px at 0% 0%, rgba(57,135,229,.22), transparent 60%), #141413; }
.ms-brand { font-size: 2rem; font-weight: 700; letter-spacing: -.02em; color: #f0efec; line-height: 1.1; }
.ms-brand span { color: #3987e5; }
.ms-tagline { color: #c3c2b7; font-size: .95rem; margin-top: 4px; }
.ms-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.ms-chip { font-size: .78rem; color: #c3c2b7; padding: 4px 11px; border-radius: 999px;
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.09); }
.ms-chip b { color: #f0efec; font-weight: 600; }
.ms-card { background: #1a1a19; border: 1px solid rgba(255,255,255,.08); border-radius: 14px;
  padding: 16px 18px; height: 100%; min-height: 138px; }
.ms-label { color: #898781; font-size: .72rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 8px; }
.ms-value { color: #f0efec; font-size: 2rem; font-weight: 650; line-height: 1.1; }
.ms-value-sm { font-size: 1.35rem; }
.ms-sub { color: #c3c2b7; font-size: .85rem; margin-top: 8px; }
.ms-meta { color: #898781; font-size: .78rem; margin-top: 6px; }
.ms-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; vertical-align: middle; }
.ms-badge { display: inline-flex; align-items: center; gap: 6px; font-size: .72rem; font-weight: 700; letter-spacing: .06em;
  color: #f0efec; padding: 3px 10px; border-radius: 999px; background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.14); vertical-align: middle; }
.ms-health { display: flex; align-items: center; gap: 16px; }
.ms-health .ms-label { white-space: nowrap; }
.ms-mix { color: #c3c2b7; font-size: .85rem; line-height: 1.7; }
.ms-gauge { width: 96px; height: 96px; flex: none; border-radius: 50%; display: grid; place-items: center;
  background: conic-gradient(var(--c) calc(var(--pct) * 1%), #2c2c2a 0); }
.ms-gauge-inner { width: 74px; height: 74px; border-radius: 50%; background: #1a1a19; display: flex;
  flex-direction: column; align-items: center; justify-content: center; }
.ms-gauge-inner b { font-size: 1.75rem; color: #f0efec; line-height: 1; }
.ms-gauge-inner span { font-size: .68rem; color: #898781; margin-top: 2px; }
.ms-feed-item { padding: 10px 14px; margin-bottom: 8px; border-radius: 10px; background: #1a1a19;
  border: 1px solid rgba(255,255,255,.06); border-left: 4px solid var(--c); }
.ms-feed-title { color: #f0efec; font-weight: 600; font-size: .92rem; display: flex; justify-content: space-between; gap: 8px; }
.ms-feed-meta { color: #898781; font-size: .78rem; margin-top: 3px; }
.ms-section { color: #f0efec; font-size: 1.05rem; font-weight: 650; margin: 6px 0 4px; }
.ms-note { color: #c3c2b7; font-size: .85rem; padding: 12px 16px; border-radius: 10px;
  background: rgba(250,178,25,.07); border: 1px solid rgba(250,178,25,.25); }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { padding: 8px 16px; font-weight: 600; }
"""

st.set_page_config(page_title="MachinaSense", page_icon="⚙️", layout="wide")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data
@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_resource
def get_splits(source, fs):
    ds = load_dataset(source, fs=fs)
    return split_chronologically(ds.df) | {"all": ds.df}


@st.cache_data(show_spinner="Analysing windows ...")
def run_pipeline(view, snr_db, drop_rate):
    bundle = get_bundle()
    df = get_splits(bundle["data_source"], bundle["fs"])[view]
    cols = bundle["sensor_cols"]
    if snr_db != "off":
        df = add_noise(df, cols, snr_db, bundle["signal_std"], seed=SEED)
    if drop_rate > 0:
        df = drop_samples(df, cols, drop_rate, seed=SEED)
    results, features = analyze(df, bundle)
    return df, results, features


def healthy_reference(bundle):
    """First healthy window of the training split, to compare alerts against."""
    train = get_splits(bundle["data_source"], bundle["fs"])["train"]
    if "label" in train and (train.label == NORMAL_LABEL).any():
        train = train[train.label == NORMAL_LABEL]
    return train.iloc[: bundle["window"]]


def load_report():
    return json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else None


# ----------------------------------------------------------------------------- helpers
def fmt_time(ts):
    """Timestamp(s) as text with millisecond precision."""
    if isinstance(ts, pd.Series):
        return ts.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
    return ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def pretty(label):
    if label == NORMAL_LABEL:
        return "Normal"
    if label == "unknown_anomaly":
        return "Unknown anomaly"
    return label.replace("_", " ").capitalize() + " fault"


def health_score(severities):
    """100 = no alerts in the recent windows; each alert costs 1/3, 2/3 or 1 by severity."""
    return 100 * (1 - pd.Series(severities).map(SEVERITY_WEIGHT).mean())


def health_state(score):
    if score >= 80:
        return "good", "Healthy"
    if score >= 50:
        return "warning", "Watch"
    if score >= 25:
        return "serious", "Degraded"
    return "critical", "Critical"


def show(markup, target=st):
    """Render HTML (joined to one line so Markdown doesn't treat indentation as code)."""
    target.markdown(" ".join(line.strip() for line in markup.strip().splitlines()), unsafe_allow_html=True)


def badge(severity):
    text = "NO ALERT" if severity == "none" else severity.upper()
    return f'<span class="ms-badge"><span class="ms-dot" style="background:{SEVERITY_COLORS[severity]}"></span>{text}</span>'


def card(label, value, sub="", meta="", small=False):
    size = " ms-value-sm" if small else ""
    return (f'<div class="ms-card"><div class="ms-label">{label}</div><div class="ms-value{size}">{value}</div>'
            f'<div class="ms-sub">{sub}</div><div class="ms-meta">{meta}</div></div>')


def health_card(score, caption):
    state, text = health_state(score)
    color = STATUS[state]
    return f"""<div class="ms-card ms-health">
        <div class="ms-gauge" style="--pct:{score:.0f}; --c:{color}">
          <div class="ms-gauge-inner"><b>{score:.0f}</b><span>of 100</span></div></div>
        <div><div class="ms-label">Machine health</div>
          <span class="ms-badge"><span class="ms-dot" style="background:{color}"></span>{text.upper()}</span>
          <div class="ms-meta">{caption}</div></div></div>"""


def status_card(row, label="Current status"):
    if row.is_alert:
        conf = f" · {row.fault_confidence:.0%} confidence" if pd.notna(row.fault_confidence) else ""
        head, sub = pretty(row.fault_type), badge(row.severity) + conf
    else:
        head, sub = "Normal operation", badge("none")
    return (f'<div class="ms-card"><div class="ms-label">{label}</div><div class="ms-value ms-value-sm">{head}</div>'
            f'<div class="ms-sub">{sub}</div><div class="ms-meta">{fmt_time(row.timestamp)}</div></div>')


def feed(alerts, limit=6):
    if alerts.empty:
        return '<div class="ms-meta">No alerts yet.</div>'
    items = []
    for i, row in alerts.iloc[::-1].head(limit).iterrows():
        lead = html.escape(row.top_features.split(",")[0])
        items.append(f'<div class="ms-feed-item" style="--c:{SEVERITY_COLORS[row.severity]}">'
                     f'<div class="ms-feed-title"><span>{pretty(row.fault_type)}</span>{badge(row.severity)}</div>'
                     f'<div class="ms-feed-meta">#{i} · {fmt_time(row.timestamp)[11:]} · {lead}</div></div>')
    return "".join(items)


# ----------------------------------------------------------------------------- charts
def score_chart(res, threshold, height=280, x_domain=None, y_domain=None):
    data = res.reset_index().rename(columns={"index": "window"})
    data["time"] = fmt_time(data.timestamp)
    data["prediction"] = data.predicted.map(pretty)
    order = ["none", "low", "medium", "high"]
    tooltip = ["window:Q", "time:N", "severity:N", "prediction:N", alt.Tooltip("anomaly_score:Q", format=".3f")]
    if "label" in data:
        tooltip.append(alt.Tooltip("label:N", title="true label"))
    points = alt.Chart(data).mark_circle(size=60, opacity=0.9).encode(
        x=alt.X("window:Q", title="window # (time order)", scale=alt.Scale(domain=x_domain) if x_domain else alt.Scale()),
        y=alt.Y("anomaly_score:Q", title="anomaly score",
                scale=alt.Scale(domain=y_domain) if y_domain else alt.Scale(zero=False)),
        color=alt.Color("severity:N", scale=alt.Scale(domain=order, range=[SEVERITY_COLORS[s] for s in order]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=tooltip)
    rule = alt.Chart(pd.DataFrame({"y": [threshold], "label": ["alert threshold"]})).mark_rule(
        strokeDash=[6, 4], strokeWidth=1.5, color=MUTED).encode(y="y:Q", tooltip=["label:N", alt.Tooltip("y:Q", format=".3f")])
    return (points + rule).properties(height=height)


def count_bars(counts, colors, height=None):
    data = counts.rename_axis("category").reset_index(name="alerts")
    base = alt.Chart(data).encode(y=alt.Y("category:N", sort=list(counts.index), title=None),
                                  x=alt.X("alerts:Q", title="alerts", axis=alt.Axis(tickMinStep=1)))
    bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
        color=alt.Color("category:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())), legend=None),
        tooltip=["category:N", "alerts:Q"])
    labels = base.mark_text(align="left", dx=6, color=INK_2).encode(text="alerts:Q")
    return (bars + labels).properties(height=height or 46 * len(data) + 50)


def waveform_chart(window_df, reference_df, channel, color, fs, height=260):
    t_ms = np.arange(len(window_df)) / fs * 1000
    frames = [pd.DataFrame({"time_ms": t_ms, "value": window_df[channel].to_numpy(), "series": "alert window"})]
    if reference_df is not None:
        frames.append(pd.DataFrame({"time_ms": t_ms[: len(reference_df)], "value": reference_df[channel].to_numpy(),
                                    "series": "healthy reference"}))
    return _overlay(frames, color, alt.X("time_ms:Q", title="time since window start (ms)"),
                    alt.Y("value:Q", title=f"{channel} signal"), height)


def spectrum(x, fs):
    x = fill_missing(np.asarray(x, float)[:, None])[:, 0]
    x = (x - x.mean()) * np.hanning(len(x))
    return np.fft.rfftfreq(len(x), 1 / fs), 20 * np.log10(np.abs(np.fft.rfft(x)) + 1e-12)


def spectrum_chart(window_df, reference_df, channel, color, fs, height=260):
    scale, unit = (1000, "kHz") if fs >= 2000 else (1, "Hz")
    frames = []
    for name, df in [("alert window", window_df), ("healthy reference", reference_df)]:
        f, db = spectrum(df[channel].to_numpy(), fs)
        frames.append(pd.DataFrame({"freq": f / scale, "value": db, "series": name}))
    return _overlay(frames, color, alt.X("freq:Q", title=f"frequency ({unit})"),
                    alt.Y("value:Q", title="magnitude (dB)", scale=alt.Scale(zero=False)), height)


def _overlay(frames, color, x, y, height):
    """Alert trace with an optional healthy reference drawn on top (it is usually the smaller one)."""
    names = [f.series.iloc[0] for f in frames]
    legend = alt.Legend(orient="bottom") if len(frames) > 1 else None   # one series: the heading names it
    color_enc = alt.Color("series:N", title=None, legend=legend,
                          scale=alt.Scale(domain=names, range=[color, MUTED][: len(names)]))
    tooltip = [alt.Tooltip(x.shorthand, format=".2f"), alt.Tooltip("value:Q", format=".4f"), "series:N"]
    layers = [alt.Chart(f).mark_line(strokeWidth=1).encode(x=x, y=y, color=color_enc, tooltip=tooltip) for f in frames]
    return alt.layer(*layers).properties(height=height)


def deviation_chart(top):
    data = top.reset_index(names="feature")
    data["sigma"] = data.deviation_sigma.abs().clip(lower=0.1)
    data["base"] = 0.1                       # log scale has no zero, so bars start at 0.1σ
    data["direction"] = np.where(data.deviation_sigma >= 0, "above healthy", "below healthy")
    domain = [0.1, max(data.sigma.max(), SEVERITY_SIGMA[1]) * 2]
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4, height=18).encode(
        y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelOverlap=False)),
        x=alt.X("sigma:Q", scale=alt.Scale(type="log", domain=domain), title="deviation from healthy (σ, log scale)"),
        x2="base:Q",
        color=alt.Color("direction:N", scale=alt.Scale(domain=["above healthy", "below healthy"], range=[SERIES[1], SERIES[0]]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["feature:N", alt.Tooltip("value:Q", format=".4g"), alt.Tooltip("healthy_mean:Q", format=".4g"),
                 alt.Tooltip("deviation_sigma:Q", format="+.1f", title="deviation (σ)")])
    bands = pd.DataFrame({"sigma": list(SEVERITY_SIGMA), "label": [f"medium ≥ {SEVERITY_SIGMA[0]}σ", f"high ≥ {SEVERITY_SIGMA[1]}σ"]})
    rules = alt.Chart(bands).mark_rule(strokeDash=[4, 4], color=MUTED).encode(x="sigma:Q", tooltip=["label:N"])
    text = alt.Chart(bands).mark_text(align="left", dx=4, dy=-6, color=MUTED, fontSize=11).encode(
        x="sigma:Q", y=alt.value(0), text="label:N")
    return (bars + rules + text).properties(height=34 * len(data) + 20)


def confusion_chart(title, classes, matrix):
    data = pd.DataFrame([(t, p, matrix[i][j]) for i, t in enumerate(classes) for j, p in enumerate(classes)],
                        columns=["true", "predicted", "count"])
    peak = max(data["count"].max(), 1)
    base = alt.Chart(data, title=alt.Title(title, anchor="start", fontSize=13)).encode(
        x=alt.X("predicted:N", sort=classes, title="predicted", axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("true:N", sort=classes, title="true"))
    cells = base.mark_rect(cornerRadius=4, stroke="#0d0d0d", strokeWidth=2).encode(
        color=alt.Color("count:Q", scale=alt.Scale(domain=[0, peak], range=["#1f2633", "#3987e5", "#cde2fb"]), legend=None),
        tooltip=["true:N", "predicted:N", "count:Q"])
    text = base.mark_text(fontSize=14, fontWeight="bold").encode(
        text="count:Q", color=alt.condition(f"datum.count > {0.6 * peak}", alt.value("#0d0d0d"), alt.value(INK)))
    return (cells + text).properties(height=max(260, 70 * len(classes)))


def robustness_chart(rob, conditions):
    series = [("baseline_f1", "Baseline (RMS z-score)"), ("iforest_no_aug_f1", "IF, trained on clean data"),
              ("iforest_f1", "IF, trained with augmentation")]
    series = [(col, name) for col, name in series if col in rob[conditions[0]]]
    names = [name for _, name in series]
    data = pd.DataFrame([{"condition": c.replace("noise ", "").replace("drop ", ""), "series": name, "f1": rob[c][col]}
                         for c in conditions for col, name in series])
    order = list(dict.fromkeys(data.condition))
    enc = dict(x=alt.X("condition:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
               y=alt.Y("f1:Q", title="detection F1", scale=alt.Scale(domain=[0.5, 1.02])),
               color=alt.Color("series:N", scale=alt.Scale(domain=names, range=SERIES[: len(names)]),
                               legend=alt.Legend(orient="bottom", title=None, direction="vertical", labelLimit=400)))
    lines = alt.Chart(data).mark_line(strokeWidth=2).encode(**enc)
    points = alt.Chart(data).mark_point(filled=True, opacity=1).encode(
        **enc, shape=alt.Shape("series:N", scale=alt.Scale(domain=names, range=["circle", "square", "triangle-up"]), legend=None),
        size=alt.Size("series:N", scale=alt.Scale(domain=names, range=[220, 110, 60]), legend=None),
        tooltip=["series:N", "condition:N", alt.Tooltip("f1:Q", format=".3f")])
    return (lines + points).properties(height=300)


# ----------------------------------------------------------------------------- page
try:
    bundle = get_bundle()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

with st.sidebar:
    st.markdown("### ⚙️ MachinaSense")
    st.caption("Fault detection & predictive maintenance")
    st.markdown("**Data**")
    view = st.radio("Windows to analyse", ["test", "val", "all"], label_visibility="collapsed",
                    format_func={"test": "Held-out test split", "val": "Validation split",
                                 "all": "Whole dataset (includes training data)"}.get)
    st.markdown("**Simulate sensor problems**")
    snr_db = st.select_slider("Added noise (SNR vs healthy signal)", ["off", 40, 30, 20, 10, 0],
                              format_func=lambda v: v if v == "off" else f"{v} dB")
    drop_rate = st.slider("Dropped samples", 0.0, 0.3, 0.0, 0.01, format="%.2f")
    with st.expander("How alerts are scored"):
        st.markdown(f"""
- **Alert**: Isolation Forest score above **{bundle['if_threshold']:.3f}** (tuned on validation data).
- **Fault type**: Random Forest trained on labelled faults.
- **Severity**: largest feature deviation from the healthy profile —
  low < {SEVERITY_SIGMA[0]}σ ≤ medium < {SEVERITY_SIGMA[1]}σ ≤ high.
- **Machine health**: 100 minus the alert load over the last {RECENT} windows
  (low ⅓, medium ⅔, high 1 per window).""")

signal_df, results, features = run_pipeline(view, snr_db, drop_rate)
has_labels = "label" in results
threshold = bundle["if_threshold"]
all_alerts = results[results.is_alert]
fault_classes = list(bundle["rf"].classes_) if bundle["rf"] is not None else ["unknown_anomaly"]
fault_colors = {pretty(c): SERIES[i % len(SERIES)] for i, c in enumerate(fault_classes)}
y_domain = [float(results.anomaly_score.min()) - 0.01, float(results.anomaly_score.max()) + 0.01]
cols = bundle["sensor_cols"]

# Header + machine summary
sim = []
if snr_db != "off":
    sim.append(f"noise {snr_db} dB")
if drop_rate > 0:
    sim.append(f"{drop_rate:.0%} dropped")
chips = [f"<b>{html.escape(str(bundle['data_source']))}</b>", f"{len(cols)} sensors · {bundle['fs']:g} Hz",
         f"window {bundle['window'] / bundle['fs'] * 1000:.0f} ms" if bundle["fs"] >= 100 else f"window {bundle['window']} samples",
         {"test": "held-out test split", "val": "validation split", "all": "whole dataset"}[view]]
if sim:
    chips.append("simulated: " + ", ".join(sim))
show(f"""<div class="ms-hero"><div><div class="ms-brand">Machina<span>Sense</span></div>
    <div class="ms-tagline">Intelligent industrial fault detection &amp; predictive maintenance</div></div>
    <div class="ms-chips">{''.join(f'<span class="ms-chip">{c}</span>' for c in chips)}</div></div>""")

latest = results.iloc[-1]
counts = all_alerts.severity.value_counts()
summary = st.columns([1.45, 1.5, 1, 1.05, 1.35])
show(health_card(health_score(results.severity.tail(RECENT)), f"last {min(RECENT, len(results))} windows"), summary[0])
show(status_card(latest, "Latest status"), summary[1])
show(card("Windows analysed", f"{len(results):,}", f"{bundle['window']} samples each"), summary[2])
show(card("Alerts", f"{len(all_alerts):,}", " ".join(f"{SEVERITY_ICONS[s]} {counts.get(s, 0)}" for s in SEVERITIES),
          f"{len(all_alerts) / max(len(results), 1):.0%} of windows"), summary[3])
mix = all_alerts.fault_type.value_counts().reindex(fault_classes).dropna().astype(int)
mix_html = "<br>".join(f'<span class="ms-dot" style="background:{fault_colors.get(pretty(f), MUTED)}"></span> '
                       f'{pretty(f).replace(" fault", "")} <b>{c}</b>' for f, c in mix.items()) or "no alerts"
show(f'<div class="ms-card"><div class="ms-label">Faults detected</div><div class="ms-mix">{mix_html}</div></div>', summary[4])
st.write("")

tab_overview, tab_live, tab_alerts, tab_eval = st.tabs(["📊 Overview", "▶️ Live replay", "🚨 Alerts & evidence", "🧪 Evaluation"])

# ----------------------------------------------------------------------------- overview
with tab_overview:
    left, right = st.columns([2.2, 1], gap="large")
    with left:
        show('<div class="ms-section">Anomaly score per window</div>')
        st.altair_chart(score_chart(results, threshold, 300), width="stretch")
        st.caption("Dashed line = alert threshold, tuned on the validation split. Hover a point for details.")
        b1, b2 = st.columns(2, gap="large")
        with b1:
            show('<div class="ms-section">Alerts by fault type</div>')
            by_fault = all_alerts.fault_type.map(pretty).value_counts().reindex(list(fault_colors), fill_value=0)
            st.altair_chart(count_bars(by_fault, fault_colors), width="stretch")
        with b2:
            show('<div class="ms-section">Alerts by severity</div>')
            by_sev = counts.reindex(SEVERITIES, fill_value=0)
            st.altair_chart(count_bars(by_sev, {s: SEVERITY_COLORS[s] for s in SEVERITIES}), width="stretch")
    with right:
        show('<div class="ms-section">Latest alerts</div>')
        show(feed(all_alerts, limit=7))

# ----------------------------------------------------------------------------- alerts & evidence
with tab_alerts:
    f1, f2 = st.columns(2)
    shown_sev = f1.multiselect("Severity", SEVERITIES, default=SEVERITIES)
    shown_fault = f2.multiselect("Predicted fault", fault_classes, default=fault_classes, format_func=pretty)
    alerts = all_alerts[all_alerts.severity.isin(shown_sev) & all_alerts.fault_type.isin(shown_fault)]
    if alerts.empty:
        st.info("No alerts for the current selection.")
    else:
        table = pd.DataFrame({
            "time": fmt_time(alerts.timestamp),
            "severity": [f"{SEVERITY_ICONS[s]} {s}" for s in alerts.severity],
            "predicted fault": alerts.fault_type.map(pretty),
            "confidence": alerts.fault_confidence,
            "anomaly score": alerts.anomaly_score,
            "top contributing features": alerts.top_features,
        }, index=alerts.index)
        if has_labels:
            table["true label (practice data)"] = alerts.label
        st.dataframe(table, width="stretch", height=260,
                     column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                    "anomaly score": st.column_config.NumberColumn(format="%.3f")})

        choice = st.selectbox("Inspect alert", alerts.index, format_func=lambda i: (
            f"#{i} · {table.loc[i, 'time']} · {alerts.severity[i]} · {pretty(alerts.fault_type[i])}"))
        alert = results.loc[choice]
        top = explain(features.loc[choice], bundle)
        lead = top.iloc[0]
        ratio = f"{lead.x_healthy:.1f}× healthy, " if pd.notna(lead.x_healthy) else ""

        e1, e2, e3, e4 = st.columns(4)
        show(card("Severity", badge(alert.severity), f"largest deviation {alert.max_deviation_sigma:.0f}σ"), e1)
        conf = f"{alert.fault_confidence:.0%} confidence" if pd.notna(alert.fault_confidence) else ""
        show(card("Predicted fault", pretty(alert.fault_type), conf, small=True), e2)
        show(card("Window", fmt_time(alert.timestamp)[11:], f"to {fmt_time(alert.end_time)[11:]}",
                  fmt_time(alert.timestamp)[:10], small=True), e3)
        show(card("Anomaly score", f"{alert.anomaly_score:.3f}", f"threshold {threshold:.3f}"), e4)
        st.markdown(f"Most abnormal feature: **{top.index[0]} = {lead.value:.4g}** ({ratio}{lead.deviation_sigma:+.0f}σ "
                    f"from the healthy mean). Severity is **{alert.severity}** because the largest deviation is "
                    f"{alert.max_deviation_sigma:.0f}σ.")

        window_df = signal_df.iloc[int(alert.start): int(alert.start) + bundle["window"]]
        reference_df = healthy_reference(bundle)
        for i, channel in enumerate(cols):
            color = SERIES[i % len(SERIES)]
            w_col, s_col = st.columns(2, gap="large")
            with w_col:
                show(f'<div class="ms-section">{html.escape(channel)} sensor · waveform</div>')
                st.altair_chart(waveform_chart(window_df, reference_df, channel, color, bundle["fs"]), width="stretch")
            with s_col:
                show(f'<div class="ms-section">{html.escape(channel)} sensor · spectrum</div>')
                st.altair_chart(spectrum_chart(window_df, reference_df, channel, color, bundle["fs"]), width="stretch")
        if snr_db != "off" or drop_rate > 0:
            st.caption("The alert window includes the simulated noise / dropped samples (gaps in the waveform); "
                       "dropped samples are interpolated before features and spectrum are computed.")

        show('<div class="ms-section">Top contributing features</div>')
        st.altair_chart(deviation_chart(top), width="stretch")
        with st.expander("Feature values"):
            st.dataframe(top.rename(columns={"value": "this window", "healthy_mean": "healthy mean", "healthy_std": "healthy std",
                                             "deviation_sigma": "deviation (σ)", "x_healthy": "× healthy"}),
                         width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.4g") for c in
                                        ["this window", "healthy mean", "healthy std", "deviation (σ)", "× healthy"]})

# ----------------------------------------------------------------------------- evaluation
with tab_eval:
    report = load_report()
    if report is None:
        st.info("No evaluation report yet - run `python evaluate.py`.")
    else:
        det, clf, e2e = report["detection"], report["classification"], report["end_to_end"]
        iso, base = det["Isolation Forest"], det["Baseline (RMS z-score)"]
        k = st.columns(4)
        show(card("Isolation Forest F1", f"{iso['f1']:.3f}", f"precision {iso['precision']:.2f} · recall {iso['recall']:.2f}",
                  f"false alarms {iso['false_alarm_rate']:.1%}"), k[0])
        show(card("Baseline F1", f"{base['f1']:.3f}", f"precision {base['precision']:.2f} · recall {base['recall']:.2f}",
                  f"false alarms {base['false_alarm_rate']:.1%}"), k[1])
        show(card("Fault type macro F1", f"{clf['macro avg']['f1']:.3f}", "Random Forest, true fault windows"), k[2])
        show(card("End-to-end macro F1", f"{e2e['macro avg']['f1']:.3f}", "normal + fault types"), k[3])
        st.write("")
        show(f"""<div class="ms-note"><b>How to read this.</b> Scores are on the held-out test split
            ({report.get('test_windows', '?')} windows of <code>{html.escape(str(report.get('data_source', '')))}</code>),
            never used for training or tuning. On CWRU each class is one short recording, so near-perfect
            scores are expected and say little about new machines - the robustness test below is the more
            honest view.</div>""")

        show('<div class="ms-section" style="margin-top:18px">Baseline vs model (detection)</div>')
        det_table = pd.DataFrame(det).T.rename(columns={"false_alarm_rate": "false alarm rate",
                                                         "end_to_end_macro_f1": "end-to-end macro F1"})
        st.dataframe(det_table, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.3f")
                                                                 for c in det_table.columns})

        show('<div class="ms-section">Confusion matrices</div>')
        cms = report.get("confusion_matrices", [])
        for row in range(0, len(cms), 2):
            for col, cm in zip(st.columns(2, gap="large"), cms[row: row + 2]):
                col.altair_chart(confusion_chart(cm["title"], cm["classes"], cm["matrix"]), width="stretch")

        rob = report["robustness"]
        show('<div class="ms-section">Robustness: test split corrupted after training</div>')
        r1, r2 = st.columns(2, gap="large")
        noise = ["clean"] + [c for c in rob if c.startswith("noise") and "+" not in c]
        drops = ["clean"] + [c for c in rob if c.startswith("drop") and "naive" not in c]
        with r1:
            st.caption("Added noise (SNR vs healthy signal)")
            st.altair_chart(robustness_chart(rob, noise), width="stretch")
        with r2:
            st.caption("Dropped samples (interpolated)")
            st.altair_chart(robustness_chart(rob, drops), width="stretch")
        with st.expander("All robustness conditions"):
            st.dataframe(pd.DataFrame(rob).T, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.3f") for c in pd.DataFrame(rob).T.columns})
        with st.expander("Per-class results"):
            c1, c2 = st.columns(2)
            c1.markdown("**Fault type (Random Forest)**")
            c1.dataframe(pd.DataFrame(clf).T, width="stretch")
            c2.markdown("**End-to-end (Isolation Forest → Random Forest)**")
            c2.dataframe(pd.DataFrame(e2e).T, width="stretch")

# ----------------------------------------------------------------------------- live replay
# Rendered last: while playing, this fragment re-runs on a timer without re-running the rest of the page.
ss = st.session_state
replay_key = (view, snr_db, drop_rate)
if ss.get("replay_key") != replay_key:
    ss.replay_key, ss.replay_idx, ss.replay_playing = replay_key, 0, False

with tab_live:
    n = len(results)
    c1, c2, c3, _ = st.columns([1, 1, 1.4, 3])
    if c1.button("⏸ Pause" if ss.replay_playing else "▶ Play", type="primary", width="stretch"):
        if not ss.replay_playing and ss.replay_idx >= n - 1:
            ss.replay_idx = 0
        ss.replay_playing = not ss.replay_playing
        st.rerun()
    if c2.button("⟲ Restart", width="stretch"):
        ss.replay_idx = 0
        st.rerun()
    speed = c3.select_slider("Speed", list(REPLAY_SPEEDS), value="1×", label_visibility="collapsed")

    @st.fragment(run_every=REPLAY_SPEEDS[speed] if ss.replay_playing else None)
    def live_replay():
        if ss.replay_playing:
            if ss.replay_idx >= n - 1:
                ss.replay_playing = False
                st.rerun()
            ss.replay_idx += 1
        i = ss.replay_idx
        upto, row = results.iloc[: i + 1], results.iloc[i]
        st.progress((i + 1) / n, text=f"Window {i + 1} of {n} · {fmt_time(row.timestamp)}"
                                      + ("" if ss.replay_playing else " · paused"))
        s = st.columns([1.35, 1.5, 1, 1])
        show(health_card(health_score(upto.severity.tail(RECENT)), f"last {min(RECENT, i + 1)} windows"), s[0])
        show(status_card(row, "This window"), s[1])
        show(card("Alerts so far", int(upto.is_alert.sum()),
                  " ".join(f"{SEVERITY_ICONS[sv]} {(upto.severity == sv).sum()}" for sv in SEVERITIES)), s[2])
        over = row.anomaly_score > threshold
        show(card("Anomaly score", f"{row.anomaly_score:.3f}", f"threshold {threshold:.3f}",
                  "above threshold → alert" if over else "below threshold"), s[3])
        st.write("")
        left, right = st.columns([2.2, 1], gap="large")
        with left:
            show('<div class="ms-section">Incoming windows</div>')
            st.altair_chart(score_chart(upto, threshold, 230, x_domain=[0, n - 1], y_domain=y_domain), width="stretch")
            window_df = signal_df.iloc[int(row.start): int(row.start) + bundle["window"]]
            for col, (j, channel) in zip(st.columns(len(cols)), enumerate(cols)):
                with col:
                    show(f'<div class="ms-section">{html.escape(channel)} sensor · current window</div>')
                    st.altair_chart(waveform_chart(window_df, None, channel, SERIES[j % len(SERIES)], bundle["fs"], 180),
                                    width="stretch")
        with right:
            show('<div class="ms-section">Alert feed</div>')
            show(feed(upto[upto.is_alert], limit=6))

    live_replay()
