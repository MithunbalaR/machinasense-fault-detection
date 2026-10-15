"""Work orders: maintenance tasks raised from incidents, tracked from Open to Done."""
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard import state
from dashboard.ui import SEVERITY_ICONS, card, hero, page_title, show
from maintenance import incidents

EDITABLE = ["status", "assignee", "notes"]


def create_for_fleet():
    """One work order per incident across the whole fleet (existing ones are skipped)."""
    bundle = state.get_bundle()
    snr_db, drop_rate = state.simulation()
    specs = state.fleet_specs(bundle["data_source"], bundle["fs"])
    created = 0
    for mid, spec in specs.items():
        _, results, _, _ = state.run_machine(mid, snr_db, drop_rate)
        for _, inc in incidents(results).iterrows():
            created += state.create_work_order(mid, spec["name"], inc)[1]
    return created


def render():
    orders = state.load_orders()
    now = datetime.now()
    for o in orders:
        o["overdue"] = o["status"] != "Done" and datetime.strptime(o["due"], "%Y-%m-%d %H:%M") < now

    hero([f"<b>{len(orders)} work orders</b>", "saved to data/work_orders.json"])
    page_title("Work orders", "Maintenance tasks raised from incidents. Assign them, track progress and add notes; "
                              "changes are saved automatically.")
    k = st.columns(4)
    show(card("Open", sum(o["status"] == "Open" for o in orders), "waiting for a technician"), k[0])
    show(card("In progress", sum(o["status"] == "In progress" for o in orders), "being worked on"), k[1])
    show(card("Overdue", sum(o["overdue"] for o in orders), "past the due date for their urgency"), k[2])
    show(card("Done", sum(o["status"] == "Done" for o in orders), "completed"), k[3])

    b1, b2, b3, _ = st.columns([1.5, 1, 1, 1.5])
    if b1.button("➕ Raise orders for all incidents", type="primary", width="stretch",
                  help="One work order per incident across the fleet; existing ones are skipped"):
        n = create_for_fleet()
        st.toast(f"Created {n} new work order(s)" if n else "All incidents already have work orders", icon="🛠")
        st.rerun()
    if b3.button("🗑 Clear completed", width="stretch", disabled=not any(o["status"] == "Done" for o in orders)):
        state.save_orders([o for o in orders if o["status"] != "Done"])
        st.rerun()

    if not orders:
        st.info("No work orders yet. Use the button above, or open a machine → **Incidents** → *Create work order*.")
        return

    df = pd.DataFrame(orders)
    b2.download_button("⬇ Export CSV", df.drop(columns=["key"]).to_csv(index=False).encode("utf-8"),
                       file_name="machinasense_work_orders.csv", mime="text/csv", width="stretch")

    f1, f2 = st.columns(2)
    show_status = f1.multiselect("Status", state.WORK_ORDER_STATUSES, default=["Open", "In progress"])
    machines = sorted(df.machine_name.unique())
    show_machines = f2.multiselect("Machine", machines, default=machines)
    view = df[df.status.isin(show_status) & df.machine_name.isin(show_machines)].copy()
    view["severity"] = [f"{SEVERITY_ICONS.get(s, '')} {s}" for s in view.severity]
    view["due"] = [("⚠ " if o else "") + d for d, o in zip(view.due, view.overdue)]
    columns = ["id", "machine_name", "fault", "severity", "detected", "due", "status", "assignee", "notes", "urgency", "action"]

    edited = st.data_editor(
        view[columns].set_index("id"), width="stretch", height=min(80 + 36 * len(view), 520),
        disabled=[c for c in columns if c not in EDITABLE and c != "id"],
        column_config={
            "machine_name": st.column_config.TextColumn("machine"),
            "status": st.column_config.SelectboxColumn("status", options=state.WORK_ORDER_STATUSES, required=True),
            "assignee": st.column_config.TextColumn("assignee", help="Technician responsible"),
            "notes": st.column_config.TextColumn("notes", width="medium"),
            "due": st.column_config.TextColumn("due", help="⚠ = overdue"),
            "action": st.column_config.TextColumn("recommended action", width="large"),
        }, key="wo_editor")

    changes = 0
    by_id = {o["id"]: o for o in orders}
    for wid, row in edited.iterrows():
        for field in EDITABLE:
            value = "" if pd.isna(row[field]) else str(row[field])
            if by_id[wid][field] != value:
                by_id[wid][field] = value
                changes += 1
    if changes:
        state.save_orders([{k: v for k, v in o.items() if k != "overdue"} for o in orders])
        st.toast(f"Saved {changes} change(s)", icon="💾")
    st.caption("Due dates follow the alert urgency: high = 24 h, medium = 7 days, low = 30 days from creation.")
