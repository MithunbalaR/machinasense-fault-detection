"""Shared look and feel: colours, CSS, and small HTML building blocks (cards, badges, feeds)."""
import html
import re
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
.ms-chip { font-size: .82rem; color: #c3c2b7; padding: 4px 11px; border-radius: 999px;
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.09); }
.ms-chip b { color: #f0efec; font-weight: 600; }
.ms-card { background: #1a1a19; border: 1px solid rgba(255,255,255,.08); border-radius: 14px;
  padding: 16px 18px; height: 100%; min-height: 138px; margin-bottom: 10px; }
.ms-label { color: #a3a29c; font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 8px; }
.ms-value { color: #f0efec; font-size: 2rem; font-weight: 650; line-height: 1.1; }
.ms-value-sm { font-size: 1.35rem; }
.ms-sub { color: #c3c2b7; font-size: .85rem; margin-top: 8px; }
.ms-meta { color: #a3a29c; font-size: .82rem; margin-top: 6px; }
.ms-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; vertical-align: middle; }
.ms-badge { display: inline-flex; align-items: center; gap: 6px; font-size: .78rem; font-weight: 700; letter-spacing: .06em;
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
.ms-gauge-inner span { font-size: .75rem; color: #a3a29c; margin-top: 2px; }
.ms-feed-item { padding: 10px 14px; margin-bottom: 8px; border-radius: 10px; background: #1a1a19;
  border: 1px solid rgba(255,255,255,.06); border-left: 4px solid var(--c); }
.ms-feed-title { color: #f0efec; font-weight: 600; font-size: .92rem; display: flex; justify-content: space-between; gap: 8px; }
.ms-feed-meta { color: #a3a29c; font-size: .82rem; margin-top: 3px; }
.ms-section { color: #f0efec; font-size: 1.05rem; font-weight: 650; margin: 6px 0 4px; }
.ms-note { color: #c3c2b7; font-size: .85rem; padding: 12px 16px; border-radius: 10px;
  background: rgba(250,178,25,.07); border: 1px solid rgba(250,178,25,.25); }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { padding: 8px 16px; font-weight: 600; }
.ms-machine { background: #1a1a19; border: 1px solid rgba(255,255,255,.08); border-radius: 14px; padding: 16px 18px;
  border-top: 4px solid var(--c); min-height: 250px; }
.ms-machine-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; flex-wrap: wrap; }
.ms-machine-name { color: #f0efec; font-size: 1.1rem; font-weight: 650; white-space: nowrap; }
.ms-machine-area { color: #a3a29c; font-size: .82rem; }
.ms-machine-body { display: flex; align-items: center; gap: 16px; margin-top: 12px; }
.ms-machine-body .ms-gauge { width: 76px; height: 76px; }
.ms-machine-body .ms-gauge-inner { width: 58px; height: 58px; }
.ms-machine-body .ms-gauge-inner b { font-size: 1.35rem; }
.ms-kv { color: #c3c2b7; font-size: .82rem; line-height: 1.65; }
.ms-kv b { color: #f0efec; font-weight: 600; }
.ms-page-title { color: #f0efec; font-size: 1.5rem; font-weight: 700; margin: 2px 0 2px; }
.ms-page-sub { color: #a3a29c; font-size: .95rem; margin-bottom: 14px; }
/* ---- accessibility ---- */
.ms-feed { list-style: none; padding: 0 !important; margin: 0 !important; }
.ms-feed > li { margin: 0 0 8px 0 !important; }
.ms-chips > li.ms-chip { margin: 0 !important; font-size: .82rem !important; line-height: 1.5; }
.ms-sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden;
  clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
.ms-skip { position: absolute; left: -9999px; top: 8px; z-index: 1000; padding: 8px 14px; border-radius: 8px;
  background: #f0efec; color: #0d0d0d; font-weight: 700; }
.ms-skip:focus { left: 16px; }
/* Visible keyboard focus on every interactive element (WCAG 2.4.7). */
button:focus-visible, a:focus-visible, input:focus-visible, textarea:focus-visible, [role="tab"]:focus-visible,
[role="slider"]:focus-visible, [role="combobox"]:focus-visible, [tabindex]:focus-visible {
  outline: 3px solid #f0efec !important; outline-offset: 2px !important; }
/* Primary buttons: white text needs a darker blue for 4.5:1 contrast (WCAG 1.4.3). */
[data-testid="stBaseButton-primary"] { background-color: #2a6ac0 !important; border-color: #2a6ac0 !important; color: #ffffff !important; }
[data-testid="stBaseButton-primary"]:hover { background-color: #22589f !important; border-color: #22589f !important; }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
"""

# Extra CSS for the user's accessibility settings (sidebar -> Accessibility).
A11Y_CSS = {
    "large_text": """html { font-size: 118%; } .ms-value { font-size: 2.2rem; } .ms-label, .ms-meta, .ms-feed-meta,
        .ms-badge, .ms-chip { font-size: .92rem; } [data-testid="stMarkdownContainer"] p { font-size: 1.05rem; }""",
    "high_contrast": """.ms-card, .ms-machine, .ms-feed-item, .ms-hero, .ms-chip, .ms-badge { border-color: #f0efec !important; }
        .ms-label, .ms-meta, .ms-feed-meta, .ms-sub, .ms-machine-area, .ms-page-sub, .ms-tagline, .ms-kv, .ms-mix,
        .ms-chip, .ms-gauge-inner span { color: #ffffff !important; }
        .ms-hero { background: #000 !important; } .ms-card, .ms-machine, .ms-feed-item { background: #000 !important; }
        [data-testid="stCaptionContainer"] { color: #ffffff !important; }""",
    "reduce_motion": "*, *::before, *::after { animation: none !important; transition: none !important; scroll-behavior: auto !important; }",
}

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


def dot(color):
    """Decorative colour dot. Hidden from screen readers: the text next to it carries the meaning."""
    return f'<span class="ms-dot" aria-hidden="true" style="background:{color}"></span>'


def badge(severity):
    text = "NO ALERT" if severity == "none" else severity.upper()
    label = "No alert" if severity == "none" else f"Severity {severity}"
    return f'<span class="ms-badge" aria-label="{label}">{dot(SEVERITY_COLORS[severity])}{text}</span>'


def card(label, value, sub="", meta="", small=False):
    size = " ms-value-sm" if small else ""
    return (f'<div class="ms-card" role="group" aria-label="{html.escape(_plain(label))}">'
            f'<div class="ms-label">{label}</div><div class="ms-value{size}">{value}</div>'
            f'<div class="ms-sub">{sub}</div><div class="ms-meta">{meta}</div></div>')


def health_card(score, caption):
    state, text = health_state(score)
    color = STATUS[state]
    return f"""<div class="ms-card ms-health" role="group" aria-label="Machine health">
        {gauge(score, color, text)}
        <div><div class="ms-label">Machine health</div>
          <span class="ms-badge">{dot(color)}{text.upper()}</span>
          <div class="ms-meta">{caption}</div></div></div>"""


def status_card(row, label="Current status"):
    if row.is_alert:
        conf = f" · {row.fault_confidence:.0%} confidence" if pd.notna(row.fault_confidence) else ""
        head, sub = pretty(row.fault_type), badge(row.severity) + conf
        meta = "→ " + recommend(row.fault_type, row.severity)["urgency"]
    else:
        head, sub, meta = "Normal operation", badge("none"), "No action needed"
    return (f'<div class="ms-card" role="group" aria-label="{html.escape(label)}">'
            f'<div class="ms-label">{label}</div><div class="ms-value ms-value-sm">{head}</div>'
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
    mark = dot(STATUS[text[0]]) + " " if text[0] else ""
    sub = (f"strongest defect peak: {pretty(best).replace(' fault', '').lower()}, {db:.0f} dB above background "
           f"(clear peak ≥ {physics.PEAK_DB} dB)")
    return card("Physics check (envelope spectrum)", mark + text[1], sub, small=True)


def feed(alerts, limit=6, label="Latest alerts, newest first"):
    if alerts.empty:
        return '<p class="ms-meta">No alerts yet.</p>'
    items = []
    for i, row in alerts.iloc[::-1].head(limit).iterrows():
        lead = html.escape(row.top_features.split(",")[0])
        title = pretty(row.fault_type) + (f" · {html.escape(row.machine)}" if "machine" in alerts.columns else "")
        items.append(f'<li class="ms-feed-item" style="--c:{SEVERITY_COLORS[row.severity]}">'
                     f'<div class="ms-feed-title"><span>{title}</span>{badge(row.severity)}</div>'
                     f'<div class="ms-feed-meta">Window {i} · {fmt_time(row.timestamp)[11:]} · {lead}</div></li>')
    return f'<ul class="ms-feed" aria-label="{label}">{"".join(items)}</ul>'




def hero(chips):
    """Brand banner with context chips (data source, settings)."""
    show(f"""<a class="ms-skip" href="#main-content">Skip to main content</a>
        <div class="ms-hero" role="banner"><div><div class="ms-brand">Machina<span>Sense</span></div>
        <div class="ms-tagline">Intelligent industrial fault detection &amp; predictive maintenance</div></div>
        <ul class="ms-chips ms-feed" aria-label="Context">{''.join(f'<li class="ms-chip">{c}</li>' for c in chips)}</ul></div>""")


def page_title(title, subtitle=""):
    show(f'<div id="main-content" class="ms-page-title" role="heading" aria-level="1" tabindex="-1">{title}</div>'
         f'<p class="ms-page-sub">{subtitle}</p>')


def section(title, target=st):
    """Section heading (level 2) so screen-reader users can jump between sections."""
    show(f'<div class="ms-section" role="heading" aria-level="2">{title}</div>', target)


def gauge(score, color, state_label=""):
    """Ring gauge, exposed to assistive technology as a meter with its value and state."""
    text = f"Health {score:.0f} of 100" + (f", {state_label}" if state_label else "")
    return (f'<div class="ms-gauge" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{score:.0f}" '
            f'aria-valuetext="{text}" aria-label="{text}" style="--pct:{score:.0f}; --c:{color}">'
            f'<div class="ms-gauge-inner" aria-hidden="true"><b>{score:.0f}</b><span>of 100</span></div></div>')


def machine_card(machine_id, spec, latest, health_now, ttc_text, n_alerts):
    """Fleet tile: health gauge, state, latest fault and time to critical."""
    state, label = health_state(health_now)
    color = STATUS[state]
    fault = pretty(latest.fault_type) if latest.is_alert else "Normal operation"
    sev = badge(latest.severity) if latest.is_alert else badge("none")
    return f"""<article class="ms-machine" style="--c:{color}" aria-label="{html.escape(spec['name'])}: {label}">
        <div class="ms-machine-name" role="heading" aria-level="3">{html.escape(spec['name'])}</div>
        <div class="ms-machine-area">{html.escape(spec['area'])} · {html.escape(machine_id)}</div>
        <div style="margin-top:8px"><span class="ms-badge">{dot(color)}{label.upper()}</span></div>
        <div class="ms-machine-body">{gauge(health_now, color, label)}
          <div class="ms-kv"><b>{fault}</b> {sev}<br>Time to critical: <b>{ttc_text}</b><br>Alerts: <b>{n_alerts}</b></div></div>
        </article>"""


def ttc_text(health, seconds_per_window):
    t = time_to_critical(health, seconds_per_window)
    if health[-1] < CRITICAL:
        return "Now"
    return "Stable" if t is None else f"≈ {format_duration(t['seconds'])}"


def _plain(markup):
    """Text content of a small HTML snippet (for aria-labels)."""
    return re.sub(r"<[^>]+>", "", str(markup)).strip()


def a11y(key):
    """State of one accessibility setting chosen in the sidebar."""
    return bool(st.session_state.get(f"a11y_{key}", False))


def show_chart(chart, summary, data=None, target=st):
    """Render a chart with a text alternative (WCAG 1.1.1).

    `summary` becomes the chart's accessible description and is always available to
    screen readers; with "Show chart data as tables" on, everyone sees it plus the data.
    """
    target.altair_chart(chart.properties(description=summary), width="stretch")
    if a11y("tables"):
        target.caption(f"Text summary: {summary}")
        if data is not None:
            with target.expander("Chart data as a table"):
                st.dataframe(data, width="stretch", height=min(40 + 35 * len(data), 320))
    else:
        show(f'<p class="ms-sr-only">Chart: {html.escape(summary)}</p>', target)
