"""Fleet overview: every monitored machine at a glance, ranked by risk."""
import html

import pandas as pd
import streamlit as st

from dashboard import nav, state
from dashboard.charts import fleet_health_chart, sparkline
from dashboard.ui import (show_chart, section, SERIES, STATUS, card, feed, hero, machine_card, page_title, pretty, show, ttc_text)
from maintenance import health_state, incidents

STATE_ORDER = {"critical": 0, "serious": 1, "warning": 2, "good": 3}


def render():
    bundle = state.get_bundle()
    snr_db, drop_rate = state.simulation()
    specs = state.fleet_specs(bundle["data_source"], bundle["fs"])
    runs = {mid: state.run_machine(mid, snr_db, drop_rate) for mid in specs}
    seconds_per_window = bundle["step"] / bundle["fs"]

    sim = [f"noise {snr_db} dB"] if snr_db != "off" else []
    sim += [f"{drop_rate:.0%} dropped"] if drop_rate > 0 else []
    hero([f"<b>{len(specs)} machines</b>", f"model: {html.escape(str(bundle['data_source']))}",
          f"{len(bundle['sensor_cols'])} sensors each"] + (["simulated: " + ", ".join(sim)] if sim else []))
    page_title("Fleet overview", "Every monitored machine at a glance. Demo fleet built from the held-out test "
                                 "recordings: one healthy machine and one per fault type.")

    # ------------------------------------------------------------------------- KPIs
    rows = []
    for mid, (_, res, _, health) in runs.items():
        key, label = health_state(health[-1])
        rows.append({"id": mid, "machine": specs[mid]["name"], "area": specs[mid]["area"], "health": round(health[-1]),
                     "state": label, "state_key": key, "time to critical": ttc_text(health, seconds_per_window),
                     "latest": pretty(res.fault_type.iloc[-1]) if res.is_alert.iloc[-1] else "Normal operation",
                     "alerts": int(res.is_alert.sum()), "incidents": len(incidents(res))})
    table = pd.DataFrame(rows).sort_values(["state_key", "health"], key=lambda c: c.map(STATE_ORDER) if c.name == "state_key" else c)
    orders = state.load_orders()
    open_orders = [o for o in orders if o["status"] != "Done"]
    k = st.columns(5)
    show(card("Machines monitored", len(specs), f"{len(bundle['sensor_cols'])} sensors each"), k[0])
    show(card("Healthy", int((table.state_key == "good").sum()), "health ≥ 80"), k[1])
    show(card("At risk", int(table.state_key.isin(["warning", "serious"]).sum()), "watch or degraded (25-80)"), k[2])
    show(card("Critical", int((table.state_key == "critical").sum()), "health < 25: act now"), k[3])
    show(card("Open work orders", len(open_orders), f"{len(orders)} in total"), k[4])

    # ------------------------------------------------------------------------- machine tiles
    section('Machines')
    colors = {specs[mid]["name"]: SERIES[i % len(SERIES)] for i, mid in enumerate(specs)}
    per_row = 4
    ids = list(specs)
    for start in range(0, len(ids), per_row):
        for col, mid in zip(st.columns(per_row, gap="medium"), ids[start:start + per_row]):
            _, res, _, health = runs[mid]
            with col:
                show(machine_card(mid, specs[mid], res.iloc[-1], health[-1], ttc_text(health, seconds_per_window),
                                  int(res.is_alert.sum())))
                show_chart(sparkline(health, STATUS[health_state(health[-1])[0]]),
                           f"Health trend of {specs[mid]['name']}: from {health[0]:.0f} to {health[-1]:.0f} over {len(health)} windows.")
                if st.button("Open machine →", key=f"open_{mid}", width="stretch",
                             help=f"Open the detail page of {specs[mid]['name']}"):
                    st.session_state.machine = mid
                    st.switch_page(nav.PAGES["machine"])

    # ------------------------------------------------------------------------- trends + feed
    left, right = st.columns([2.2, 1], gap="large")
    with left:
        section('Health of every machine over time')
        show_chart(fleet_health_chart({specs[m]["name"]: runs[m][3] for m in specs}, colors),
                   "Health index over time for each machine. Current health: "
                   + ", ".join(f"{specs[m]['name']} {runs[m][3][-1]:.0f}" for m in specs) + ".",
                   # machines can have different lengths, so align them as Series (shorter ones get blanks)
                   pd.DataFrame({specs[m]["name"]: pd.Series(runs[m][3].round(1)) for m in specs}).rename_axis("window"))
        section('Risk ranking')
        st.dataframe(table.drop(columns=["state_key"]).set_index("id"), width="stretch",
                     column_config={"health": st.column_config.ProgressColumn("health", min_value=0, max_value=100, format="%d")})
    with right:
        section('Latest alerts across the fleet')
        fleet_alerts = pd.concat([runs[m][1][runs[m][1].is_alert].assign(machine=specs[m]["name"]) for m in specs])
        show(feed(fleet_alerts.sort_values("timestamp"), limit=8, label="Latest alerts across the fleet, newest first"))
