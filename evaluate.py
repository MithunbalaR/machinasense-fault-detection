"""Evaluate the saved models on the held-out test split.

1. Detection (normal vs abnormal): RMS z-score baseline vs Isolation Forest
2. Fault classification: Random Forest on the truly faulty test windows
3. End-to-end (normal / fault type), and severity counts per true class
4. Robustness: the same test split with added noise and randomly dropped samples
   (models are trained on clean data - this mimics a degraded sensor in the field)
5. Physics check: share of alerts whose envelope spectrum confirms the predicted fault
6. Generalisation (CWRU only): the 0 HP model on unseen motor loads (1-3 HP)

Writes reports/metrics.md, reports/metrics.json, reports/confusion_matrices.png
and reports/robustness.png.

Usage:
    python evaluate.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

import physics
from data_loader import CWRU_LOADS, NORMAL_LABEL, load_cwru, load_dataset
from features import add_noise, drop_samples, split_chronologically
from model import SEED, analyze, load_bundle, train

REPORT_DIR = Path(__file__).parent / "reports"
SNR_LEVELS_DB = [30, 20, 10, 0]   # noise relative to a healthy machine's vibration level
DROP_RATES = [0.01, 0.05, 0.20]

# Chart colours (validated categorical slots 1-3, sequential blue ramp, recessive ink).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])


# ----------------------------------------------------------------------------- metrics
def detection_metrics(labels, alerts):
    abnormal = labels != NORMAL_LABEL
    p, r, f, _ = precision_recall_fscore_support(abnormal, alerts, average="binary", zero_division=0)
    false_alarm = alerts[~abnormal].mean() if (~abnormal).any() else np.nan
    return {"precision": p, "recall": r, "f1": f, "false_alarm_rate": false_alarm}


def per_class_table(labels, predicted, classes):
    p, r, f, n = precision_recall_fscore_support(labels, predicted, labels=classes, zero_division=0)
    table = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": n}, index=classes)
    table.loc["macro avg"] = [p.mean(), r.mean(), f.mean(), n.sum()]
    return table


def md_table(df, floatfmt="{:.3f}"):
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- plots
def plot_confusion_matrices(panels, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4))
    for ax, (title, labels, predicted, classes) in zip(np.atleast_1d(axes), panels):
        cm = confusion_matrix(labels, predicted, labels=classes)
        ax.imshow(cm, cmap=BLUES, vmin=0, vmax=max(cm.max(), 1))
        for i in range(len(classes)):
            for j in range(len(classes)):
                dark = cm[i, j] > 0.55 * cm.max()
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                        color="#fcfcfb" if dark else INK)
        ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(classes)), classes, fontsize=9)
        ax.set_xlabel("predicted", color=MUTED)
        ax.set_ylabel("true", color=MUTED)
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_robustness(rob, path):
    series = [("baseline_f1", "Baseline (RMS z-score)"),
              ("iforest_no_aug_f1", "Isolation Forest, trained on clean data only"),
              ("iforest_f1", "Isolation Forest, trained with noise/drop augmentation")]
    series = [(col, name) for col, name in series if col in rob]
    markers = [("o", 11), ("s", 8), ("^", 6)]   # big -> small, so overlapping points stay visible
    groups = [("Added noise (SNR vs healthy signal)", ["clean"] + [f"noise {s} dB" for s in SNR_LEVELS_DB]),
              ("Dropped samples (interpolated)", ["clean"] + [f"drop {r:.0%}" for r in DROP_RATES])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (title, conditions) in zip(axes, groups):
        sub = rob.loc[conditions]
        x = np.arange(len(conditions))
        for color, (marker, size), (col, name) in zip(SERIES, markers, series):
            ax.plot(x, sub[col], color=color, linewidth=2, marker=marker, markersize=size,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.2, label=name)
        ax.set_xticks(x, [c.replace("noise ", "").replace("drop ", "") for c in conditions])
        ax.set_ylim(0.5, 1.03)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
    axes[0].set_ylabel("detection F1 on held-out test windows", color=MUTED)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(names), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def robustness_test(test_df, bundle, clean_bundle=None):
    """Score every corrupted version of the test split. `clean_bundle` (optional) is
    the same pipeline trained without augmentation, to show what augmentation buys."""
    cols, ref = bundle["sensor_cols"], bundle["signal_std"]
    conditions = {"clean": test_df}
    for snr in SNR_LEVELS_DB:
        conditions[f"noise {snr} dB"] = add_noise(test_df, cols, snr, ref, seed=SEED)
    for rate in DROP_RATES:
        conditions[f"drop {rate:.0%}"] = drop_samples(test_df, cols, rate, seed=SEED)
    conditions[f"drop {DROP_RATES[-1]:.0%} zero-filled (naive)"] = drop_samples(test_df, cols, DROP_RATES[-1], seed=SEED, fill="zero")
    conditions["noise 20 dB + drop 5%"] = drop_samples(add_noise(test_df, cols, 20, ref, seed=SEED), cols, 0.05, seed=SEED)

    rows = {}
    for name, df in conditions.items():
        res, _ = analyze(df, bundle)
        labels = res.label.to_numpy()
        base, iso = detection_metrics(labels, res.baseline_alert.to_numpy()), detection_metrics(labels, res.is_alert.to_numpy())
        row = {"baseline_f1": base["f1"], "baseline_false_alarm": base["false_alarm_rate"],
               "iforest_f1": iso["f1"], "iforest_false_alarm": iso["false_alarm_rate"],
               "end_to_end_macro_f1": f1_score(labels, res.predicted, average="macro", zero_division=0)}
        if clean_bundle is not None:
            res0, _ = analyze(df, clean_bundle)
            iso0 = detection_metrics(labels, res0.is_alert.to_numpy())
            row.update({"iforest_no_aug_f1": iso0["f1"], "iforest_no_aug_false_alarm": iso0["false_alarm_rate"],
                        "end_to_end_no_aug_macro_f1": f1_score(labels, res0.predicted, average="macro", zero_division=0)})
        rows[name] = row
    rob = pd.DataFrame.from_dict(rows, orient="index")
    rob.index.name = "condition"
    print(rob.round(3).to_string())
    return rob


def physics_agreement(test_df, res, bundle):
    """Physics verdict per true class for every alert (None if defect frequencies are unknown)."""
    if not physics.available(bundle.get("physics"), bundle["fs"]):
        return None
    alerts = res[res.is_alert]
    verdicts = [physics.check(test_df, int(r.start), bundle["window"], bundle["fs"], bundle["physics"], r.fault_type)["verdict"]
                for _, r in alerts.iterrows()]
    table = pd.crosstab(alerts.label, pd.Series(verdicts, index=alerts.index, name="physics"))
    table = table.reindex(columns=["confirms", "inconclusive", "disagrees"], fill_value=0)
    table.index.name = "true label"
    return table


def generalisation(bundle):
    """CWRU only: score the 0 HP model on every window of the unseen 1-3 HP recordings."""
    if bundle["data_source"] != "cwru":
        return None
    rows = {}
    for hp in sorted(CWRU_LOADS)[1:]:
        res, _ = analyze(load_cwru(hp).df, bundle)
        labels = res.label.to_numpy()
        faulty = labels != NORMAL_LABEL
        iso, base = detection_metrics(labels, res.is_alert.to_numpy()), detection_metrics(labels, res.baseline_alert.to_numpy())
        rows[f"{hp} HP"] = {
            "windows": len(res),
            "iforest_f1": iso["f1"], "iforest_false_alarm": iso["false_alarm_rate"],
            "baseline_f1": base["f1"], "baseline_false_alarm": base["false_alarm_rate"],
            "fault_type_accuracy": (res.fault_type.to_numpy()[faulty] == labels[faulty]).mean(),
            "end_to_end_macro_f1": f1_score(labels, res.predicted, average="macro", zero_division=0),
        }
        print(f"  {hp} HP: {rows[f'{hp} HP']}")
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "motor load (unseen)"
    return table


def main():
    bundle = load_bundle()
    if not bundle["has_labels"]:
        print("The data has no labels, so precision/recall/F1 cannot be computed.")
        return
    REPORT_DIR.mkdir(exist_ok=True)
    ds = load_dataset(bundle["data_source"], fs=bundle["fs"])
    test_df = split_chronologically(ds.df)["test"]
    res, _ = analyze(test_df, bundle)
    labels = res.label.to_numpy()
    fault_classes = sorted(set(labels) - {NORMAL_LABEL})
    all_classes = [NORMAL_LABEL] + fault_classes
    print(f"held-out test split: {len(res)} windows from {bundle['data_source']}\n")

    # 1. detection side by side
    detection = pd.DataFrame({
        "Baseline (RMS z-score)": detection_metrics(labels, res.baseline_alert.to_numpy()),
        "Isolation Forest": detection_metrics(labels, res.is_alert.to_numpy()),
    }).T
    detection["end_to_end_macro_f1"] = [f1_score(labels, res.baseline_predicted, average="macro", zero_division=0),
                                        f1_score(labels, res.predicted, average="macro", zero_division=0)]
    detection.index.name = "detector (+ RF for fault type)"
    print("Detection (abnormal = positive class):\n" + detection.round(3).to_string() + "\n")

    # 2. fault classification on truly faulty windows
    faulty = labels != NORMAL_LABEL
    classification = per_class_table(labels[faulty], res.fault_type.to_numpy()[faulty], fault_classes)
    print("Fault classification (Random Forest, true fault windows only):\n" + classification.round(3).to_string() + "\n")

    # 3. end-to-end + severity
    end_to_end = per_class_table(labels, res.predicted.to_numpy(), all_classes)
    print("End-to-end (Isolation Forest -> Random Forest):\n" + end_to_end.round(3).to_string() + "\n")
    severity = pd.crosstab(res.label, res.severity).reindex(index=all_classes, columns=["none", "low", "medium", "high"], fill_value=0)
    severity.index.name = "true label"
    print("Severity of alerts per true class:\n" + severity.to_string() + "\n")

    panels = [
        ("Detection: baseline", np.where(faulty, "abnormal", "normal"), np.where(res.baseline_alert, "abnormal", "normal"), ["normal", "abnormal"]),
        ("Detection: Isolation Forest", np.where(faulty, "abnormal", "normal"), np.where(res.is_alert, "abnormal", "normal"), ["normal", "abnormal"]),
        ("Fault type: Random Forest", labels[faulty], res.fault_type.to_numpy()[faulty], fault_classes),
        ("End-to-end: IF -> RF", labels, res.predicted.to_numpy(), all_classes),
    ]
    plot_confusion_matrices(panels, REPORT_DIR / "confusion_matrices.png")

    # 4. robustness
    print("Robustness (models trained on the training split, test split corrupted):")
    clean_bundle = None
    if bundle.get("augment_copies"):
        print("(also training the same pipeline without augmentation for comparison)")
        clean_bundle, _, _ = train(bundle["data_source"], bundle["fs"], bundle["window"], bundle["step"],
                                   bundle["n_bands"], augment_copies=0)
    rob = robustness_test(test_df, bundle, clean_bundle)
    plot_robustness(rob, REPORT_DIR / "robustness.png")

    # 5. physics check, 6. unseen motor loads
    phys = physics_agreement(test_df, res, bundle)
    if phys is not None:
        print("\nPhysics check of test alerts (envelope spectrum vs predicted fault):\n" + phys.to_string())
    print("\nGeneralisation to unseen motor loads:")
    gen = generalisation(bundle)
    if gen is not None:
        print(gen.round(3).to_string())

    md = [f"# MachinaSense evaluation\n\nData: `{bundle['data_source']}`, held-out test split "
          f"({len(res)} windows of {bundle['window']} samples). Thresholds were tuned on the validation split only.\n",
          "## Detection: baseline vs model\n", md_table(detection), "",
          "## Fault classification (Random Forest, true fault windows)\n", md_table(classification), "",
          "## End-to-end (Isolation Forest -> Random Forest)\n", md_table(end_to_end), "",
          "## Severity of alerts per true class\n", md_table(severity, "{}"), "",
          "![Confusion matrices for baseline detection, Isolation Forest detection, Random Forest fault type and "
          "the end-to-end system on the held-out test windows](confusion_matrices.png)\n",
          "## Robustness to noise and dropped samples\n",
          "Models are trained on the training split only; the test split is corrupted. Noise SNR is "
          "relative to the healthy machine's signal level (0 dB = noise as strong as the healthy vibration). "
          "Dropped samples are linearly interpolated by the pipeline, except the 'zero-filled (naive)' row. "
          "`no_aug` columns are the same pipeline trained without noise/drop augmentation.\n",
          md_table(rob), "", "![Line charts of detection F1 against added noise and dropped samples for the baseline "
          "and the Isolation Forest with and without augmentation](robustness.png)\n"]
    if phys is not None:
        md += ["## Physics check of alerts\n",
               "Envelope spectrum of ~0.34 s around each alert: does the peak at a bearing defect frequency "
               f"match the predicted fault? A clear peak is at least {physics.PEAK_DB} dB above the background.\n",
               md_table(phys, "{}"), ""]
    if gen is not None:
        md += ["## Generalisation to unseen motor loads\n",
               "The model trained on 0 HP data, scored on every window of the 1-3 HP recordings "
               "(different shaft speed and load, never seen in training or tuning).\n", md_table(gen), ""]
    (REPORT_DIR / "metrics.md").write_text("\n".join(md), encoding="utf-8")
    metrics = {"detection": detection.to_dict(orient="index"), "classification": classification.to_dict(orient="index"),
               "end_to_end": end_to_end.to_dict(orient="index"), "severity": severity.to_dict(orient="index"),
               "robustness": rob.to_dict(orient="index"),
               "confusion_matrices": [{"title": title, "classes": classes,
                                       "matrix": confusion_matrix(true, pred, labels=classes).tolist()}
                                      for title, true, pred, classes in panels],
               "test_windows": len(res), "data_source": bundle["data_source"],
               "physics_check": phys.to_dict(orient="index") if phys is not None else None,
               "generalisation": gen.to_dict(orient="index") if gen is not None else None}
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {REPORT_DIR / 'metrics.md'} (+ .json, confusion_matrices.png, robustness.png)")


if __name__ == "__main__":
    main()
