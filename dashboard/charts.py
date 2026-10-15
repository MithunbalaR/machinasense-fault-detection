"""Altair charts used across the dashboard pages."""
import altair as alt
import numpy as np
import pandas as pd

import physics
from dashboard.ui import INK, INK_2, MUTED, SERIES, SEVERITIES, SEVERITY_COLORS, STATUS, fmt_time, pretty
from features import fill_missing
from maintenance import CRITICAL, time_to_critical
from model import SEVERITY_SIGMA


def score_chart(res, threshold, height=280, x_domain=None, y_domain=None):
    data = res.reset_index().rename(columns={"index": "window"})
    data["time"] = fmt_time(data.timestamp)
    data["prediction"] = data.predicted.map(pretty)
    order = ["none", "low", "medium", "high"]
    tooltip = ["window:Q", "time:N", "severity:N", "prediction:N", alt.Tooltip("anomaly_score:Q", format=".3f")]
    if "label" in data:
        tooltip.append(alt.Tooltip("label:N", title="true label"))
    # Severity is shown by colour AND shape, so it never relies on colour alone (WCAG 1.4.1).
    points = alt.Chart(data).mark_point(size=70, opacity=0.95, filled=True).encode(
        x=alt.X("window:Q", title="window # (time order)", scale=alt.Scale(domain=x_domain) if x_domain else alt.Scale()),
        y=alt.Y("anomaly_score:Q", title="anomaly score",
                scale=alt.Scale(domain=y_domain) if y_domain else alt.Scale(zero=False)),
        color=alt.Color("severity:N", scale=alt.Scale(domain=order, range=[SEVERITY_COLORS[s] for s in order]),
                        legend=alt.Legend(orient="top", title=None)),
        shape=alt.Shape("severity:N", scale=alt.Scale(domain=order, range=["circle", "triangle-up", "diamond", "square"]),
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


def health_chart(health, seconds_per_window, x_domain=None, height=230):
    """Health index per window, the critical line, and the projected trend (if falling)."""
    data = pd.DataFrame({"window": np.arange(len(health)), "health": health})
    x = alt.X("window:Q", title="window # (time order)", scale=alt.Scale(domain=x_domain) if x_domain else alt.Scale())
    y = alt.Y("health:Q", title="health index", scale=alt.Scale(domain=[0, 100]))
    layers = [alt.Chart(data).mark_line(strokeWidth=2, color=SERIES[0]).encode(
        x=x, y=y, tooltip=["window:Q", alt.Tooltip("health:Q", format=".0f")])]
    t = time_to_critical(health, seconds_per_window)
    if t is not None:
        last = len(health) - 1
        proj = pd.DataFrame({"window": [last, last + t["windows"]], "health": [health[-1], CRITICAL]})
        layers.append(alt.Chart(proj).mark_line(strokeDash=[5, 4], strokeWidth=2, color=STATUS["serious"], clip=True).encode(
            x=x, y=y, tooltip=[alt.Tooltip("window:Q", format=".0f", title="projected critical at window")]))
    crit = pd.DataFrame({"health": [CRITICAL], "label": [f"critical ({CRITICAL})"]})
    layers.append(alt.Chart(crit).mark_rule(strokeDash=[2, 3], color=STATUS["critical"]).encode(y="health:Q"))
    layers.append(alt.Chart(crit).mark_text(align="left", dx=4, dy=-7, color=MUTED, fontSize=11).encode(
        y="health:Q", x=alt.value(0), text="label:N"))
    return alt.layer(*layers).properties(height=height)


def envelope_chart(result, fault_hz, colors):
    """Envelope spectrum in dB above background, with the bearing defect frequencies marked."""
    freqs, amp = result["freqs"], result["amp"]
    top = 4 * max(fault_hz.values())
    keep = (freqs > 5) & (freqs < top)
    background = np.median(amp[keep]) + 1e-12
    data = pd.DataFrame({"freq": freqs[keep], "db": 20 * np.log10(amp[keep] / background + 1e-12)})
    line = alt.Chart(data).mark_line(strokeWidth=1.2, color=INK_2).encode(
        x=alt.X("freq:Q", title="envelope frequency (Hz)", scale=alt.Scale(domain=[0, top])),
        y=alt.Y("db:Q", title="dB above background"),
        tooltip=[alt.Tooltip("freq:Q", format=".1f"), alt.Tooltip("db:Q", format=".1f")])
    marks = pd.DataFrame([{"freq": h * f, "fault": pretty(k), "harmonic": f"{h}×", "label": f"{pretty(k).replace(' fault', '')} {f:.0f} Hz"}
                          for k, f in fault_hz.items() for h in physics.HARMONICS])
    rules = alt.Chart(marks).mark_rule(strokeDash=[4, 3], strokeWidth=1.5).encode(
        x="freq:Q", color=alt.Color("fault:N", title="defect frequency",
                                    scale=alt.Scale(domain=list(colors), range=list(colors.values())),
                                    legend=alt.Legend(orient="bottom")),
        tooltip=["fault:N", "harmonic:N", alt.Tooltip("freq:Q", format=".1f")])
    threshold = alt.Chart(pd.DataFrame({"db": [physics.PEAK_DB]})).mark_rule(color=MUTED, strokeDash=[2, 3]).encode(y="db:Q")
    return (line + rules + threshold).properties(height=280)


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




def sparkline(health, color, height=60):
    """Tiny health trend for a fleet tile (no axes; hover for values)."""
    data = pd.DataFrame({"window": np.arange(len(health)), "health": health})
    return alt.Chart(data).mark_area(line={"color": color, "strokeWidth": 2}, color=color, opacity=0.18).encode(
        x=alt.X("window:Q", axis=None), y=alt.Y("health:Q", axis=None, scale=alt.Scale(domain=[0, 100])),
        tooltip=["window:Q", alt.Tooltip("health:Q", format=".0f")]).properties(height=height)


def fleet_health_chart(healths, colors, height=280):
    """Health index of every machine over time, with the critical level. Machines often share the
    same health (100 or 0), so each is its own layer, widest first, keeping overlapped lines visible."""
    color = alt.Color("machine:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())),
                      legend=alt.Legend(orient="bottom", title=None))
    layers = []
    for i, (name, h) in enumerate(healths.items()):
        data = pd.DataFrame({"window": np.arange(len(h)), "health": h, "machine": name})
        layers.append(alt.Chart(data).mark_line(strokeWidth=max(7 - 1.7 * i, 1.5)).encode(
            x=alt.X("window:Q", title="window # (time order)"),
            y=alt.Y("health:Q", title="health index", scale=alt.Scale(domain=[0, 100])),
            color=color, tooltip=["machine:N", "window:Q", alt.Tooltip("health:Q", format=".0f")]))
    crit = alt.Chart(pd.DataFrame({"health": [CRITICAL]})).mark_rule(strokeDash=[2, 3], color=STATUS["critical"]).encode(y="health:Q")
    return alt.layer(*layers, crit).properties(height=height)


def threshold_curve_chart(curve, tuned, chosen, height=300):
    """Precision, recall and F1 against the alert threshold, with the tuned and chosen thresholds marked."""
    data = curve.melt(id_vars="threshold", value_vars=["precision", "recall", "f1", "false alarm rate"],
                      var_name="metric", value_name="value")
    names = ["precision", "recall", "f1", "false alarm rate"]
    lines = alt.Chart(data).mark_line(strokeWidth=2).encode(
        x=alt.X("threshold:Q", title="alert threshold (anomaly score)", scale=alt.Scale(zero=False)),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1.02])),
        color=alt.Color("metric:N", scale=alt.Scale(domain=names, range=SERIES[:4]), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["metric:N", alt.Tooltip("threshold:Q", format=".3f"), alt.Tooltip("value:Q", format=".3f")])
    marks = pd.DataFrame({"threshold": [tuned, chosen], "label": ["tuned on validation", "your threshold"]})
    rules = alt.Chart(marks).mark_rule(strokeWidth=1.5).encode(
        x="threshold:Q", strokeDash=alt.StrokeDash("label:N", scale=alt.Scale(domain=list(marks.label), range=[[6, 4], [1, 0]]),
                                                    legend=alt.Legend(orient="bottom", title=None)),
        color=alt.value(INK_2), tooltip=["label:N", alt.Tooltip("threshold:Q", format=".3f")])
    return (lines + rules).properties(height=height)


def importance_chart(importances, top=15):
    data = importances.sort_values(ascending=False).head(top).rename_axis("feature").reset_index(name="importance")
    base = alt.Chart(data).encode(y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelOverlap=False)),
                                  x=alt.X("importance:Q", title="Random Forest importance (mean decrease in impurity)"))
    bars = base.mark_bar(cornerRadiusEnd=4, height=16, color=SERIES[0]).encode(
        tooltip=["feature:N", alt.Tooltip("importance:Q", format=".3f")])
    text = base.mark_text(align="left", dx=4, color=INK_2, fontSize=11).encode(text=alt.Text("importance:Q", format=".2f"))
    return (bars + text).properties(height=26 * len(data) + 40)
