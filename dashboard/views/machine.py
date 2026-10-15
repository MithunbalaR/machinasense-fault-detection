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
from dashboard.ui import (MUTED, REPLAY_SPEEDS, SERIES, SEVERITIES, SEVERITY_COLORS, SEVERITY_ICONS, badge, card, feed,
                          fmt_time, health_card, hero, physics_card, pretty, show, status_card, ttc_card)
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
    mix_html = "<br>".join(f'<span class="ms-dot" style="background:{fault_colors.get(pretty(f), MUTED)}"></span> '
                           f'{pretty(f).replace(" fault", "")} <b>{c}</b>' for f, c in mix.items()) or "no alerts"
    show(f'<div class="ms-card"><div class="ms-label">Faults detected</div><div class="ms-mix">{mix_html}</div></div>', summary[4])
    st.write("")


    tab_overview, tab_live, tab_alerts, tab_incidents = st.tabs(
        ["📊 Overview", "▶️ Live replay", "🚨 Alerts & evidence", f"🧾 Incidents ({len(incident_table)})"])

    # ----------------------------------------------------------------------------- overview
    with tab_overview:
        left, right = st.columns([2.2, 1], gap="large")
        with left:
            show('<div class="ms-section">Anomaly score per window</div>')
            st.altair_chart(score_chart(results, threshold, 300), width="stretch")
            st.caption("Dashed line = alert threshold, tuned on the validation split. Hover a point for details.")
            show('<div class="ms-section">Machine health trend</div>')
            st.altair_chart(health_chart(health, seconds_per_window), width="stretch")
            st.caption(f"Health index = 100 minus the alert load of the last {RECENT} windows. While health is falling, "
                       "the orange dashed line projects the trend to the critical level (predictive maintenance). "
                       "CWRU recordings last seconds, so projections are in seconds; on run-to-failure data the same "
                       "projection is in hours or days.")
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
                    show(f'<div class="ms-section">{html.escape(channel)} sensor · waveform</div>')
                    st.altair_chart(waveform_chart(window_df, reference_df, channel, color, bundle["fs"]), width="stretch")
                with s_col:
                    show(f'<div class="ms-section">{html.escape(channel)} sensor · spectrum</div>')
                    st.altair_chart(spectrum_chart(window_df, reference_df, channel, color, bundle["fs"]), width="stretch")
            if snr_db != "off" or drop_rate > 0:
                st.caption("The alert window includes the simulated noise / dropped samples (gaps in the waveform); "
                           "dropped samples are interpolated before features and spectrum are computed.")

            if check is not None:
                show(f'<div class="ms-section">{html.escape(bundle["physics"]["channel"])} sensor · envelope spectrum '
                     f'(physics check)</div>')
                st.altair_chart(envelope_chart(check, bundle["physics"]["fault_hz"],
                                               {pretty(k): fault_colors.get(pretty(k), MUTED) for k in bundle["physics"]["fault_hz"]}),
                                width="stretch")
                st.caption(f"Envelope of ~{physics.CONTEXT_SAMPLES / bundle['fs']:.2f} s around the alert. A damaged bearing "
                           "rings at a defect frequency set by its geometry and the shaft speed "
                           f"({bundle['physics']['shaft_hz'] * 60:.0f} rpm); dashed lines mark each defect frequency and its "
                           f"2nd harmonic. A peak more than {physics.PEAK_DB} dB above the background (grey line) is clear evidence.")

            show('<div class="ms-section">Top contributing features</div>')
            st.altair_chart(deviation_chart(top), width="stretch")
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
                show('<div class="ms-section">Incoming windows</div>')
                st.altair_chart(score_chart(upto, threshold, 200, x_domain=[0, n - 1], y_domain=y_domain), width="stretch")
                show('<div class="ms-section">Health trend and projection</div>')
                st.altair_chart(health_chart(health_so_far, seconds_per_window, x_domain=[0, n - 1], height=200),
                                width="stretch")
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
