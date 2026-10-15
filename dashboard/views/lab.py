"""Model lab: model card, threshold playground, feature importance and the held-out evaluation."""
import html

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_fscore_support

import physics
from dashboard import state
from dashboard.charts import confusion_chart, importance_chart, robustness_chart, threshold_curve_chart
from dashboard.ui import show_chart, section, SERIES, card, hero, page_title, show
from data_loader import NORMAL_LABEL
from model import SEVERITY_SIGMA


def detection_at(scores, abnormal, threshold):
    alerts = scores > threshold
    p, r, f, _ = precision_recall_fscore_support(abnormal, alerts, average="binary", zero_division=0)
    return {"precision": p, "recall": r, "f1": f, "false alarm rate": alerts[~abnormal].mean() if (~abnormal).any() else 0.0}


def render():
    bundle = state.get_bundle()
    hero([f"<b>{html.escape(str(bundle['data_source']))}</b>", f"{len(bundle['feature_names'])} features",
          f"{bundle.get('augment_copies', 0)} augmented copies"])
    page_title("Model lab", "What the models are, how the alert threshold trades misses for false alarms, "
                            "and how they score on held-out data.")

    # ------------------------------------------------------------------------- model card
    classes = list(bundle["rf"].classes_) if bundle["rf"] is not None else []
    k = st.columns(4)
    show(card("Training data", html.escape(str(bundle["data_source"])),
              f"{len(bundle['sensor_cols'])} sensors at {bundle['fs']:g} Hz",
              "chronological 60 / 20 / 20 split per recording", small=True), k[0])
    show(card("Features", len(bundle["feature_names"]),
              f"windows of {bundle['window']} samples, step {bundle['step']}",
              f"{bundle.get('n_bands', 0)} FFT bands per sensor"), k[1])
    show(card("Detectors", "Isolation Forest", f"alert if score > {bundle['if_threshold']:.3f}",
              f"baseline: RMS |z| > {bundle['baseline'].threshold:.1f}", small=True), k[2])
    show(card("Fault classifier", "Random Forest" if classes else "none (no labels)",
              ", ".join(c.replace("_", " ") for c in classes) or "anomaly-only mode",
              f"severity bands {SEVERITY_SIGMA[0]}σ / {SEVERITY_SIGMA[1]}σ", small=True), k[3])

    tab_play, tab_imp, tab_eval = st.tabs(["🎚 Threshold playground", "🧠 Feature importance", "🧪 Held-out evaluation"])

    # ------------------------------------------------------------------------- threshold playground
    with tab_play:
        val = state.run_split("val")
        if "label" not in val:
            st.info("The data has no labels, so the precision/recall trade-off cannot be shown.")
        else:
            abnormal = (val.label != NORMAL_LABEL).to_numpy()
            scores = val.anomaly_score.to_numpy()
            tuned = float(bundle["if_threshold"])
            lo, hi = float(scores.min()), float(scores.max())
            chosen = st.slider("Alert threshold", lo, hi, min(max(tuned, lo), hi), (hi - lo) / 200, format="%.3f",
                               help="Windows with an anomaly score above this value raise an alert.")
            m = detection_at(scores, abnormal, chosen)
            c = st.columns(4)
            show(card("Precision", f"{m['precision']:.2f}", "share of alerts that are real faults"), c[0])
            show(card("Recall", f"{m['recall']:.2f}", "share of faulty windows caught"), c[1])
            show(card("F1", f"{m['f1']:.2f}", "balance of the two"), c[2])
            show(card("False alarm rate", f"{m['false alarm rate']:.0%}", "healthy windows flagged"), c[3])
            grid = np.linspace(lo, hi, 120)
            curve = pd.DataFrame([{"threshold": t, **detection_at(scores, abnormal, t)} for t in grid])
            show_chart(threshold_curve_chart(curve, tuned, chosen),
                       f"Precision, recall, F1 and false-alarm rate on the validation split for alert thresholds from "
                       f"{lo:.3f} to {hi:.3f}. At {chosen:.3f}: precision {m['precision']:.2f}, recall {m['recall']:.2f}, "
                       f"F1 {m['f1']:.2f}, false alarms {m['false alarm rate']:.0%}. Tuned threshold: {tuned:.3f}.",
                       curve.round(3).iloc[::6])
            st.caption(f"Validation split ({len(val)} windows), never the test split. Moving the threshold right means "
                       "fewer false alarms but more missed faults. The tuned threshold sits in the middle of the "
                       "best-F1 range for safety margin. This slider is for exploration and does not change the saved model.")

    # ------------------------------------------------------------------------- feature importance
    with tab_imp:
        if bundle["rf"] is None:
            st.info("No fault classifier (the data has no labels).")
        else:
            imp = pd.Series(bundle["rf"].feature_importances_, index=bundle["feature_names"])
            top3 = imp.sort_values(ascending=False).head(3)
            show_chart(importance_chart(imp), "Most important features for telling fault types apart: "
                       + ", ".join(f"{k} ({v:.2f})" for k, v in top3.items()) + ".",
                       imp.sort_values(ascending=False).head(15).rename("importance").to_frame().round(3))
            st.caption("Which features the Random Forest relies on to tell the fault types apart. "
                       "Per-alert explanations (deviation from healthy) are on the machine page.")

    # ------------------------------------------------------------------------- held-out evaluation
    with tab_eval:
        report = state.load_report()
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

            section('Baseline vs model (detection)')
            det_table = pd.DataFrame(det).T.rename(columns={"false_alarm_rate": "false alarm rate",
                                                             "end_to_end_macro_f1": "end-to-end macro F1"})
            st.dataframe(det_table, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.3f")
                                                                     for c in det_table.columns})

            section('Confusion matrices')
            cms = report.get("confusion_matrices", [])
            for row in range(0, len(cms), 2):
                for col, cm in zip(st.columns(2, gap="large"), cms[row: row + 2]):
                    matrix = pd.DataFrame(cm["matrix"], index=[f"true {c}" for c in cm["classes"]],
                                          columns=[f"predicted {c}" for c in cm["classes"]])
                    correct = int(np.trace(np.array(cm["matrix"])))
                    show_chart(confusion_chart(cm["title"], cm["classes"], cm["matrix"]),
                               f"Confusion matrix, {cm['title']}: {correct} of {int(np.sum(cm['matrix']))} windows on the "
                               "diagonal (correct). " + "; ".join(f"true {t}: " + ", ".join(
                                   f"{v} predicted {p}" for p, v in zip(cm["classes"], row) if v)
                                   for t, row in zip(cm["classes"], cm["matrix"])) + ".", matrix, target=col)

            rob = report["robustness"]
            section('Robustness: test split corrupted after training')
            r1, r2 = st.columns(2, gap="large")
            noise = ["clean"] + [c for c in rob if c.startswith("noise") and "+" not in c]
            drops = ["clean"] + [c for c in rob if c.startswith("drop") and "naive" not in c]
            with r1:
                st.caption("Added noise (SNR vs healthy signal)")
                show_chart(robustness_chart(rob, noise), "Detection F1 with added noise: " + "; ".join(
                    f"{c}: baseline {rob[c]['baseline_f1']:.2f}, Isolation Forest {rob[c]['iforest_f1']:.2f}" for c in noise) + ".",
                    pd.DataFrame(rob).T.loc[noise].round(3))
            with r2:
                st.caption("Dropped samples (interpolated)")
                show_chart(robustness_chart(rob, drops), "Detection F1 with dropped samples: " + "; ".join(
                    f"{c}: baseline {rob[c]['baseline_f1']:.2f}, Isolation Forest {rob[c]['iforest_f1']:.2f}" for c in drops) + ".",
                    pd.DataFrame(rob).T.loc[drops].round(3))
            with st.expander("All robustness conditions"):
                st.dataframe(pd.DataFrame(rob).T, width="stretch",
                             column_config={c: st.column_config.NumberColumn(format="%.3f") for c in pd.DataFrame(rob).T.columns})
            gen = report.get("generalisation")
            if gen:
                section('Generalisation: the 0 HP model on unseen motor loads')
                st.caption("Every window of the 1–3 HP recordings: a different shaft speed and load that was never used "
                           "for training or tuning. This is the most honest estimate of performance on a new operating point.")
                gen_df = pd.DataFrame(gen).T
                gen_long = gen_df.reset_index(names="load").melt(
                    id_vars="load", value_vars=["baseline_f1", "iforest_f1", "fault_type_accuracy", "end_to_end_macro_f1"],
                    var_name="metric", value_name="score")
                names = {"baseline_f1": "Baseline detection F1", "iforest_f1": "Isolation Forest detection F1",
                         "fault_type_accuracy": "Fault-type accuracy", "end_to_end_macro_f1": "End-to-end macro F1"}
                gen_long["metric"] = gen_long.metric.map(names)
                bars = alt.Chart(gen_long).mark_bar(cornerRadiusEnd=4).encode(
                    x=alt.X("load:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset=alt.XOffset("metric:N", sort=list(names.values())),
                    y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1]), title="score"),
                    color=alt.Color("metric:N", sort=list(names.values()), scale=alt.Scale(range=SERIES[:4]),
                                    legend=alt.Legend(orient="bottom", title=None, labelLimit=300)),
                    tooltip=["load:N", "metric:N", alt.Tooltip("score:Q", format=".3f")])
                show_chart(bars.properties(height=300), "Scores of the 0 HP model on unseen motor loads: " + "; ".join(
                    f"{load}: baseline F1 {r['baseline_f1']:.2f}, Isolation Forest F1 {r['iforest_f1']:.2f}, fault-type "
                    f"accuracy {r['fault_type_accuracy']:.2f}" for load, r in gen_df.iterrows()) + ".", gen_df.round(3))
                with st.expander("Generalisation table"):
                    st.dataframe(gen_df, width="stretch",
                                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in gen_df.columns if c != "windows"})
            phys = report.get("physics_check")
            if phys:
                section('Physics check of test alerts')
                st.caption(f"Does the envelope spectrum around each alert show a clear (≥ {physics.PEAK_DB} dB) peak at the "
                           "defect frequency of the predicted fault? Independent of the ML models.")
                st.table(pd.DataFrame(phys).T)
            with st.expander("Per-class results"):
                c1, c2 = st.columns(2)
                c1.markdown("**Fault type (Random Forest)**")
                c1.dataframe(pd.DataFrame(clf).T, width="stretch")
                c2.markdown("**End-to-end (Isolation Forest → Random Forest)**")
                c2.dataframe(pd.DataFrame(e2e).T, width="stretch")

