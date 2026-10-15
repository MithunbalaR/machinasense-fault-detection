"""Machine detail: health, live replay, alerts with evidence and incidents for one machine."""
import html

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import physics
from dashboard import state
from dashboard.charts import (count_bars, deviation_chart, envelope_chart, health_chart, score_chart, spectrum_chart,
                              waveform_chart)
from dashboard.ui import (section, MUTED, REPLAY_SPEEDS, SERIES, SEVERITIES, SEVERITY_COLORS, SEVERITY_ICONS, badge, card, feed,
                          a11y, dot, fmt_time, health_card, hero, physics_card, pretty, show, show_chart, status_card,
                          ttc_card, ttc_text)
from maintenance import RECENT, incidents, recommend
from model import explain


def render():
    bundle = state.get_bundle()
    snr_db, drop_rate = state.simulation()
    specs = state.fleet_specs(bundle["data_source"], bundle["fs"])
    ids = list(specs)
    pick = st.selectbox("Machine", ids, index=ids.index(state.selected_machine(specs)),
                        format_func=lambda m: f"{specs[m]['name']} · {specs[m]['area']}")
    st.session_state.machine = pick
    spec = specs[pick]
    signal_df, results, features, health = state.run_machine(pick, snr_db, drop_rate)

    has_labels = "label" in results
    threshold = bundle["if_threshold"]
    all_alerts = results[results.is_alert]
    fault_classes = list(bundle["rf"].classes_) if bundle["rf"] is not None else ["unknown_anomaly"]
    fault_colors = {pretty(c): SERIES[i % len(SERIES)] for i, c in enumerate(fault_classes)}
    y_domain = [float(results.anomaly_score.min()) - 0.01, float(results.anomaly_score.max()) + 0.01]
    cols = bundle["sensor_cols"]
    seconds_per_window = bundle["step"] / bundle["fs"]
    physics_on = physics.available(bundle.get("physics"), bundle["fs"])
    incident_table = incidents(results)

    sim = []
    if snr_db != "off":
        sim.append(f"noise {snr_db} dB")
    if drop_rate > 0:
        sim.append(f"{drop_rate:.0%} dropped")
    chips = [f"<b>{html.escape(spec['name'])}</b>", html.escape(spec["area"]), f"{len(cols)} sensors · {bundle['fs']:g} Hz",
             f"{len(results):,} windows"] + (["simulated: " + ", ".join(sim)] if sim else [])
    hero(chips)
    st.caption(f"Demo scenario (practice data): {spec['scenario']}.")

    latest = results.iloc[-1]
    counts = all_alerts.severity.value_counts()
    summary = st.columns([1.45, 1.5, 1, 1.05, 1.35])
    show(health_card(health[-1], f"last {min(RECENT, len(results))} windows"), summary[0])
    show(status_card(latest, "Latest status"), summary[1])
    show(ttc_card(health, seconds_per_window), summary[2])
    show(card("Alerts", f"{len(all_alerts):,}", " ".join(f"{SEVERITY_ICONS[s]} {counts.get(s, 0)}" for s in SEVERITIES),
              f"{len(all_alerts) / max(len(results), 1):.0%} of windows"), summary[3])
    mix = all_alerts.fault_type.value_counts().reindex(fault_classes).dropna().astype(int)
    mix_html = "<br>".join(f'{dot(fault_colors.get(pretty(f), MUTED))} '
                           f'{pretty(f).replace(" fault", "")} <b>{c}</b>' for f, c in mix.items()) or "no alerts"
    show(f'<div class="ms-card" role="group" aria-label="Faults detected"><div class="ms-label">Faults detected</div>'
         f'<div class="ms-mix">{mix_html}</div></div>', summary[4])
    st.write("")


    tab_overview, tab_live, tab_alerts, tab_incidents = st.tabs(
        ["📊 Overview", "▶️ Live replay", "🚨 Alerts & evidence", f"🧾 Incidents ({len(incident_table)})"])

    # ----------------------------------------------------------------------------- overview
    with tab_overview:
        left, right = st.columns([2.2, 1], gap="large")
        with left:
            section('Anomaly score per window')
            show_chart(score_chart(results, threshold, 300),
                       f"Anomaly score for {len(results)} windows of {spec['name']}: {len(all_alerts)} are above the "
                       f"alert threshold {threshold:.3f} (high {counts.get('high', 0)}, medium {counts.get('medium', 0)}, "
                       f"low {counts.get('low', 0)}).",
                       results[["timestamp", "anomaly_score", "severity", "fault_type"]].assign(timestamp=lambda d: fmt_time(d.timestamp)))
            st.caption("Dashed line = alert threshold, tuned on the validation split. Hover a point for details.")
            section('Machine health trend')
            show_chart(health_chart(health, seconds_per_window),
                       f"Health index of {spec['name']} over {len(health)} windows: starts at {health[0]:.0f}, lowest "
                       f"{health.min():.0f}, now {health[-1]:.0f} of 100. Time to critical: {ttc_text(health, seconds_per_window)}.",
                       pd.DataFrame({"health": health.round(1)}).rename_axis("window"))
            st.caption(f"Health index = 100 minus the alert load of the last {RECENT} windows. While health is falling, "
                       "the orange dashed line projects the trend to the critical level (predictive maintenance). "
                       "CWRU recordings last seconds, so projections are in seconds; on run-to-failure data the same "
                       "projection is in hours or days.")
            b1, b2 = st.columns(2, gap="large")
            with b1:
                section('Alerts by fault type')
                by_fault = all_alerts.fault_type.map(pretty).value_counts().reindex(list(fault_colors), fill_value=0)
                show_chart(count_bars(by_fault, fault_colors),
                           "Alerts by fault type: " + ", ".join(f"{k} {v}" for k, v in by_fault.items()) + ".",
                           by_fault.rename("alerts").to_frame())
            with b2:
                section('Alerts by severity')
                by_sev = counts.reindex(SEVERITIES, fill_value=0)
                show_chart(count_bars(by_sev, {s: SEVERITY_COLORS[s] for s in SEVERITIES}),
                           "Alerts by severity: " + ", ".join(f"{k} {v}" for k, v in by_sev.items()) + ".",
                           by_sev.rename("alerts").to_frame())
        with right:
            section('Latest alerts')
            show(feed(all_alerts, limit=7, label=f"Latest alerts on {spec['name']}, newest first"))

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
            recs = [recommend(f, s) for f, s in zip(alerts.fault_type, alerts.severity)]
            alert_report = table.assign(severity=alerts.severity, urgency=[r["urgency"] for r in recs],
                                  recommended_action=[r["action"] for r in recs])
            st.download_button("⬇ Download alert report (CSV)", alert_report.to_csv(index_label="window").encode("utf-8"),
                               file_name="machinasense_alerts.csv", mime="text/csv")
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

            rec = recommend(alert.fault_type, alert.severity)
            check = physics.check(signal_df, int(alert.start), bundle["window"], bundle["fs"], bundle["physics"],
                                  alert.fault_type) if physics_on else None
            r1, r2 = st.columns([1.4, 1])
            show(card("Recommended action", rec["urgency"], rec["action"], small=True), r1)
            show(physics_card(check), r2)

            window_df = signal_df.iloc[int(alert.start): int(alert.start) + bundle["window"]]
            reference_df = state.healthy_reference(bundle)
            for i, channel in enumerate(cols):
                color = SERIES[i % len(SERIES)]
                w_col, s_col = st.columns(2, gap="large")
                with w_col:
                    section(f'{html.escape(channel)} sensor · waveform')
                    rms = lambda x: float(np.sqrt(np.nanmean(np.square(x))))  # noqa: E731
                    show_chart(waveform_chart(window_df, reference_df, channel, color, bundle["fs"]),
                               f"{channel} vibration during the alert window ({len(window_df)} samples) drawn over a healthy "
                               f"reference window. RMS {rms(window_df[channel]):.3g} versus {rms(reference_df[channel]):.3g} "
                               f"when healthy ({rms(window_df[channel]) / max(rms(reference_df[channel]), 1e-12):.1f} times).")
                with s_col:
                    section(f'{html.escape(channel)} sensor · spectrum')
                    show_chart(spectrum_chart(window_df, reference_df, channel, color, bundle["fs"]),
                               f"Frequency spectrum of the {channel} alert window against the healthy reference, "
                               f"from 0 to {bundle['fs'] / 2:g} Hz.")
            if snr_db != "off" or drop_rate > 0:
                st.caption("The alert window includes the simulated noise / dropped samples (gaps in the waveform); "
                           "dropped samples are interpolated before features and spectrum are computed.")

            if check is not None:
                section(f'{html.escape(bundle["physics"]["channel"])} sensor · envelope spectrum (physics check)')
                peaks = ", ".join(f"{pretty(k).replace(' fault', '').lower()} {f:.0f} Hz: {check['prominence'][k]:.0f} dB"
                                  for k, f in bundle["physics"]["fault_hz"].items())
                show_chart(envelope_chart(check, bundle["physics"]["fault_hz"],
                                          {pretty(k): fault_colors.get(pretty(k), MUTED) for k in bundle["physics"]["fault_hz"]}),
                           f"Envelope spectrum around the alert. Peak height at each bearing defect frequency: {peaks}. "
                           f"Verdict: physics {check['verdict']} the predicted {pretty(alert.fault_type).lower()}.")
                st.caption(f"Envelope of ~{physics.CONTEXT_SAMPLES / bundle['fs']:.2f} s around the alert. A damaged bearing "
                           "rings at a defect frequency set by its geometry and the shaft speed "
                           f"({bundle['physics']['shaft_hz'] * 60:.0f} rpm); dashed lines mark each defect frequency and its "
                           f"2nd harmonic. A peak more than {physics.PEAK_DB} dB above the background (grey line) is clear evidence.")

            section('Top contributing features')
            show_chart(deviation_chart(top), "Features that deviate most from the healthy profile: "
                       + ", ".join(f"{name} {z:+.0f}σ" for name, z in top.deviation_sigma.head(5).items()) + ".")
            with st.expander("Feature values"):
                st.dataframe(top.rename(columns={"value": "this window", "healthy_mean": "healthy mean", "healthy_std": "healthy std",
                                                 "deviation_sigma": "deviation (σ)", "x_healthy": "× healthy"}),
                             width="stretch",
                             column_config={c: st.column_config.NumberColumn(format="%.4g") for c in
                                            ["this window", "healthy mean", "healthy std", "deviation (σ)", "× healthy"]})


    # ------------------------------------------------------------------------- incidents
    with tab_incidents:
        render_incidents(pick, spec, incident_table)

    # ----------------------------------------------------------------------------- live replay
    # Rendered last: while playing, this fragment re-runs on a timer without re-running the rest of the page.
    ss = st.session_state
    replay_key = (pick, snr_db, drop_rate)
    if ss.get("replay_key") != replay_key:
        ss.replay_key, ss.replay_idx, ss.replay_playing = replay_key, 0, False

    with tab_live:
        n = len(results)
        c1, c2, c3, c4, _ = st.columns([1, 1, 1, 1.4, 2])
        if c1.button("⏸ Pause" if ss.replay_playing else "▶ Play", type="primary", width="stretch"):
            if not ss.replay_playing and ss.replay_idx >= n - 1:
                ss.replay_idx = 0
            ss.replay_playing = not ss.replay_playing
            st.rerun()
        if c2.button("Step ▸", width="stretch", help="Advance one window at a time, at your own pace"):
            ss.replay_playing = False
            ss.replay_idx = min(ss.replay_idx + 1, n - 1)
        if c3.button("⟲ Restart", width="stretch"):
            ss.replay_idx = 0
            st.rerun()
        speed = c4.select_slider("Replay speed", list(REPLAY_SPEEDS), value="0.5×" if a11y("reduce_motion") else "1×")

        @st.fragment(run_every=REPLAY_SPEEDS[speed] if ss.replay_playing else None)
        def live_replay():
            if ss.replay_playing:
                if ss.replay_idx >= n - 1:
                    ss.replay_playing = False
                    st.rerun()
                ss.replay_idx += 1
            i = ss.replay_idx
            upto, row = results.iloc[: i + 1], results.iloc[i]
            previous = results.iloc[i - 1] if i else None
            if ss.replay_playing and row.is_alert and (previous is None or previous.severity != row.severity
                                                      or previous.fault_type != row.fault_type):
                icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}[row.severity]
                st.toast(f"**{row.severity.upper()} alert** on {spec['name']}: {pretty(row.fault_type)}", icon=icon)
            st.progress((i + 1) / n, text=f"Window {i + 1} of {n} · {fmt_time(row.timestamp)}"
                                          + ("" if ss.replay_playing else " · paused"))
            health_so_far = health[: i + 1]
            s = st.columns([1.35, 1.6, 1.1, 1])
            show(health_card(health_so_far[-1], f"last {min(RECENT, i + 1)} windows"), s[0])
            show(status_card(row, "This window"), s[1])
            show(ttc_card(health_so_far, seconds_per_window), s[2])
            show(card("Alerts so far", int(upto.is_alert.sum()),
                      " ".join(f"{SEVERITY_ICONS[sv]} {(upto.severity == sv).sum()}" for sv in SEVERITIES),
                      f"score {row.anomaly_score:.3f} vs threshold {threshold:.3f}"), s[3])
            st.write("")
            left, right = st.columns([2.2, 1], gap="large")
            with left:
                section('Incoming windows')
                show_chart(score_chart(upto, threshold, 200, x_domain=[0, n - 1], y_domain=y_domain),
                           f"Anomaly scores of the first {i + 1} of {n} windows; {int(upto.is_alert.sum())} alerts so far.")
                section('Health trend and projection')
                show_chart(health_chart(health_so_far, seconds_per_window, x_domain=[0, n - 1], height=200),
                           f"Health {health_so_far[-1]:.0f} of 100 after {i + 1} windows; time to critical "
                           f"{ttc_text(health_so_far, seconds_per_window)}.")
                window_df = signal_df.iloc[int(row.start): int(row.start) + bundle["window"]]
                for col, (j, channel) in zip(st.columns(len(cols)), enumerate(cols)):
                    with col:
                        section(f'{html.escape(channel)} sensor · current window')
                        show_chart(waveform_chart(window_df, None, channel, SERIES[j % len(SERIES)], bundle["fs"], 180),
                                   f"{channel} vibration of window {i + 1}.")
            with right:
                section('Alert feed')
                show(feed(upto[upto.is_alert], limit=6, label="Alerts so far in the replay, newest first"))

        live_replay()


def render_incidents(machine_id, spec, incident_table):
    st.caption("Consecutive alerts with the same predicted fault are grouped into one incident "
               "(gaps of up to 2 quiet windows allowed). Raise one work order per incident.")
    if incident_table.empty:
        st.info("No incidents for this machine.")
        return
    existing = {o["key"]: o for o in state.load_orders()}
    for i, inc in incident_table.iterrows():
        rec = recommend(inc.fault_type, inc.severity)
        key = f"{machine_id}|{inc.start_time}|{inc.fault_type}"
        c1, c2 = st.columns([4, 1.2], vertical_alignment="center")
        show(f'''<div class="ms-feed-item" style="--c:{SEVERITY_COLORS[inc.severity]}">
            <div class="ms-feed-title"><span>{pretty(inc.fault_type)} · {inc.windows} windows</span>{badge(inc.severity)}</div>
            <div class="ms-feed-meta">{fmt_time(inc.start_time)} → {fmt_time(inc.end_time)[11:]} · peak deviation {inc.peak_sigma:.0f}σ</div>
            <div class="ms-feed-meta">→ {rec["urgency"]} {rec["action"]}</div></div>''', c1)
        if key in existing:
            c2.success(f"{existing[key]['id']} · {existing[key]['status']}")
        elif c2.button("Create work order", key=f"wo_{machine_id}_{i}", width="stretch"):
            order, _ = state.create_work_order(machine_id, spec["name"], inc)
            st.toast(f"Created {order['id']} for {spec['name']}", icon="🛠")
            st.rerun()
