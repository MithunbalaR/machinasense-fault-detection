"""Shared look and feel: colours, CSS, and small HTML building blocks (cards, badges, feeds)."""
import html
from pathlib import Path

import pandas as pd
import streamlit as st

import physics
from data_loader import NORMAL_LABEL
from maintenance import CRITICAL, TREND_WINDOWS, format_duration, health_state, recommend, time_to_critical

REPORT_PATH = Path(__file__).parent.parent / "reports" / "metrics.json"
SEVERITIES = ["high", "medium", "low"]
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
  padding: 16px 18px; height: 100%; min-height: 138px; margin-bottom: 10px; }
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
.ms-machine { background: #1a1a19; border: 1px solid rgba(255,255,255,.08); border-radius: 14px; padding: 16px 18px;
  border-top: 4px solid var(--c); min-height: 250px; }
.ms-machine-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; flex-wrap: wrap; }
.ms-machine-name { color: #f0efec; font-size: 1.1rem; font-weight: 650; white-space: nowrap; }
.ms-machine-area { color: #898781; font-size: .78rem; }
.ms-machine-body { display: flex; align-items: center; gap: 16px; margin-top: 12px; }
.ms-machine-body .ms-gauge { width: 76px; height: 76px; }
.ms-machine-body .ms-gauge-inner { width: 58px; height: 58px; }
.ms-machine-body .ms-gauge-inner b { font-size: 1.35rem; }
.ms-kv { color: #c3c2b7; font-size: .82rem; line-height: 1.65; }
.ms-kv b { color: #f0efec; font-weight: 600; }
.ms-page-title { color: #f0efec; font-size: 1.5rem; font-weight: 700; margin: 2px 0 2px; }
.ms-page-sub { color: #898781; font-size: .9rem; margin-bottom: 14px; }
"""

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
        {gauge(score, color)}
        <div><div class="ms-label">Machine health</div>
          <span class="ms-badge"><span class="ms-dot" style="background:{color}"></span>{text.upper()}</span>
          <div class="ms-meta">{caption}</div></div></div>"""


def status_card(row, label="Current status"):
    if row.is_alert:
        conf = f" · {row.fault_confidence:.0%} confidence" if pd.notna(row.fault_confidence) else ""
        head, sub = pretty(row.fault_type), badge(row.severity) + conf
        meta = "→ " + recommend(row.fault_type, row.severity)["urgency"]
    else:
        head, sub, meta = "Normal operation", badge("none"), "No action needed"
    return (f'<div class="ms-card"><div class="ms-label">{label}</div><div class="ms-value ms-value-sm">{head}</div>'
            f'<div class="ms-sub">{sub}</div><div class="ms-meta">{meta}<br>{fmt_time(row.timestamp)}</div></div>')


def ttc_card(health, seconds_per_window, label="Time to critical"):
    """Health-trend projection (maintenance.time_to_critical) as a card."""
    t = time_to_critical(health, seconds_per_window)
    if health[-1] < CRITICAL:
        value, sub = "Now", "health is already critical"
    elif t is None:
        value, sub = "Stable", "health is not trending down"
    else:
        value, sub = f"≈ {format_duration(t['seconds'])}", f"≈ {t['windows']:.0f} windows at the current trend"
    return card(label, value, sub, f"straight-line trend of the last {TREND_WINDOWS} windows", small=True)


def physics_card(result):
    if result is None:
        return card("Physics check", "Not available", "bearing defect frequencies unknown", small=True)
    best, db = result["best"], result["prominence"][result["best"]]
    text = {"confirms": ("good", "Confirms the ML prediction"),
            "disagrees": ("warning", f"Suggests {pretty(best).lower()} instead"),
            "inconclusive": (None, "Inconclusive")}[result["verdict"]]
    dot = f'<span class="ms-dot" style="background:{STATUS[text[0]]}"></span> ' if text[0] else ""
    sub = (f"strongest defect peak: {pretty(best).replace(' fault', '').lower()}, {db:.0f} dB above background "
           f"(clear peak ≥ {physics.PEAK_DB} dB)")
    return card("Physics check (envelope spectrum)", dot + text[1], sub, small=True)


def feed(alerts, limit=6):
    if alerts.empty:
        return '<div class="ms-meta">No alerts yet.</div>'
    items = []
    for i, row in alerts.iloc[::-1].head(limit).iterrows():
        lead = html.escape(row.top_features.split(",")[0])
        title = pretty(row.fault_type) + (f" · {html.escape(row.machine)}" if "machine" in alerts.columns else "")
        items.append(f'<div class="ms-feed-item" style="--c:{SEVERITY_COLORS[row.severity]}">'
                     f'<div class="ms-feed-title"><span>{title}</span>{badge(row.severity)}</div>'
                     f'<div class="ms-feed-meta">#{i} · {fmt_time(row.timestamp)[11:]} · {lead}</div></div>')
    return "".join(items)




def hero(chips):
    """Brand banner with context chips (data source, settings)."""
    show(f"""<div class="ms-hero"><div><div class="ms-brand">Machina<span>Sense</span></div>
        <div class="ms-tagline">Intelligent industrial fault detection &amp; predictive maintenance</div></div>
        <div class="ms-chips">{''.join(f'<span class="ms-chip">{c}</span>' for c in chips)}</div></div>""")


def page_title(title, subtitle=""):
    show(f'<div class="ms-page-title">{title}</div><div class="ms-page-sub">{subtitle}</div>')


def gauge(score, color):
    return (f'<div class="ms-gauge" style="--pct:{score:.0f}; --c:{color}"><div class="ms-gauge-inner">'
            f'<b>{score:.0f}</b><span>of 100</span></div></div>')


def machine_card(machine_id, spec, latest, health_now, ttc_text, n_alerts):
    """Fleet tile: health gauge, state, latest fault and time to critical."""
    state, label = health_state(health_now)
    color = STATUS[state]
    fault = pretty(latest.fault_type) if latest.is_alert else "Normal operation"
    sev = badge(latest.severity) if latest.is_alert else badge("none")
    return f"""<div class="ms-machine" style="--c:{color}">
        <div class="ms-machine-name">{html.escape(spec['name'])}</div>
        <div class="ms-machine-area">{html.escape(spec['area'])} · {html.escape(machine_id)}</div>
        <div style="margin-top:8px"><span class="ms-badge"><span class="ms-dot" style="background:{color}"></span>{label.upper()}</span></div>
        <div class="ms-machine-body">{gauge(health_now, color)}
          <div class="ms-kv"><b>{fault}</b> {sev}<br>Time to critical: <b>{ttc_text}</b><br>Alerts: <b>{n_alerts}</b></div></div>
        </div>"""


def ttc_text(health, seconds_per_window):
    t = time_to_critical(health, seconds_per_window)
    if health[-1] < CRITICAL:
        return "Now"
    return "Stable" if t is None else f"≈ {format_duration(t['seconds'])}"
