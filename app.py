"""MachinaSense dashboard (multi-page Streamlit app).

Run:  streamlit run app.py
      (after `python model.py`; `python evaluate.py` fills the Model lab's evaluation)

Pages live in dashboard/views/, shared pieces in dashboard/ (ui, charts, state).
"""
import streamlit as st

st.set_page_config(page_title="MachinaSense", page_icon="⚙️", layout="wide")

from dashboard import nav, state  # noqa: E402  (set_page_config must come first)
from dashboard.ui import CSS  # noqa: E402
from dashboard.views import fleet, lab, machine, studio, work_orders  # noqa: E402
from maintenance import RECENT  # noqa: E402
from model import SEVERITY_SIGMA  # noqa: E402

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

nav.PAGES.update({
    "fleet": st.Page(fleet.render, title="Fleet overview", icon="🏭", url_path="fleet", default=True),
    "machine": st.Page(machine.render, title="Machine detail", icon="⚙️", url_path="machine"),
    "work_orders": st.Page(work_orders.render, title="Work orders", icon="🛠", url_path="work-orders"),
    "studio": st.Page(studio.render, title="Data studio", icon="📤", url_path="data-studio"),
    "lab": st.Page(lab.render, title="Model lab", icon="🧪", url_path="model-lab"),
})
page = st.navigation({"Monitor": [nav.PAGES["fleet"], nav.PAGES["machine"], nav.PAGES["work_orders"]],
                      "Build": [nav.PAGES["studio"], nav.PAGES["lab"]]})

try:
    bundle = state.get_bundle()
except FileNotFoundError as err:
    st.error(f"{err}. Train a model first: `python model.py`.")
    st.stop()

with st.sidebar:
    st.markdown("### ⚙️ MachinaSense")
    st.caption("Fault detection & predictive maintenance")
    st.markdown("**Simulate sensor problems**")
    st.select_slider("Added noise (SNR vs healthy signal)", ["off", 40, 30, 20, 10, 0], key="sim_snr",
                     format_func=lambda v: v if v == "off" else f"{v} dB")
    st.slider("Dropped samples", 0.0, 0.3, 0.0, 0.01, format="%.2f", key="sim_drop")
    st.caption("Applies to every page: the models see the corrupted signal.")
    with st.expander("How alerts are scored"):
        st.markdown(f"""
- **Alert**: Isolation Forest score above **{bundle['if_threshold']:.3f}** (tuned on validation data).
- **Fault type**: Random Forest trained on labelled faults.
- **Severity**: largest feature deviation from the healthy profile —
  low < {SEVERITY_SIGMA[0]}σ ≤ medium < {SEVERITY_SIGMA[1]}σ ≤ high.
- **Machine health**: 100 minus the alert load over the last {RECENT} windows
  (low ⅓, medium ⅔, high 1 per window).
- **Physics check**: envelope-spectrum peak at the bearing defect frequency.""")

page.run()
