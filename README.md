# MachinaSense: industrial fault detection (PS 05)

A small, end-to-end pipeline that ingests time-stamped sensor data from one type of
equipment, flags abnormal windows, classifies the fault type, and shows every alert
with its evidence, timestamp and severity. It is practised on the CWRU bearing
dataset and built so that a CSV of `timestamp, sensors..., [label]` can be swapped in
on the day.

```
data_loader.py  -> features.py -> baseline.py / model.py -> evaluate.py
 (only file that   (split, windows,  (detectors +            (held-out metrics,
  knows formats)    features)         classifier, saved)      robustness test)
                                              \-> app.py (Streamlit dashboard)
```

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run each step

| Step | Command | What it does |
|---|---|---|
| 1. Data | `python data_loader.py` | Downloads the CWRU subset to `data/raw/` and prints a summary |
| 2. Train | `python model.py` | Trains the baseline, Isolation Forest and Random Forest; tunes thresholds on validation; saves `models/machinasense.joblib` |
| 3. Evaluate | `python evaluate.py` | Held-out test metrics, baseline vs model, robustness test; writes `reports/` |
| 4. Dashboard | `streamlit run app.py` | Multi-page app: fleet overview, machine detail with live replay, work orders, data studio, model lab |
| 5. Tests | `python -m pytest tests` | Accessibility checks (WCAG contrast, text alternatives, labels, semantics) |

The whole pipeline takes well under a minute on a laptop. Seeds are fixed (`SEED = 42`), so
re-running gives identical numbers.

## How it works

**Data (`data_loader.py`).** Everything is converted to one table: one row per sample with
`timestamp`, one column per sensor, `label` (if known) and `segment` (a contiguous
recording). The CWRU subset is the 48 kHz drive-end experiments at 0 HP load
(~1797 rpm) with 0.007" faults: `normal` (97.mat), `inner_race` (109), `ball` (122),
`outer_race` (135), using the drive-end (DE) and fan-end (FE) accelerometers. All four
files share one sample rate, so the model can't tell "normal" apart just by sample rate.
CWRU has no clock, so timestamps are synthetic.

**Split (`features.py`).** Every recording is cut **by time** into 60% train, 20%
validation and 20% test. Windows overlap, so shuffling windows randomly would leak
near-copies of the test set into training. The test split is used only by `evaluate.py`.

**Features (`features.py`).** Windows of 2048 samples (43 ms, about 1.3 shaft
revolutions) with 50% overlap. Per sensor: mean, RMS, peak, crest factor, kurtosis,
skewness, and the share of spectral energy in 8 equal FFT bands (0–3 kHz, 3–6 kHz, …).
Dropped samples (NaN) are linearly interpolated before features are computed.

**Baseline (`baseline.py`).** z-score of window RMS against healthy training windows;
alert if `|z| > threshold` on any sensor.

**Models (`model.py`).**
- *Isolation Forest* is trained on healthy windows only, so it needs no fault labels and
  can flag fault types it has never seen.
- *Random Forest* is trained on labelled fault windows and names the fault type for each alert.
- *System output:* "normal" unless the Isolation Forest fires, then the Random Forest's fault type.
- *Thresholds:* both detectors use the threshold that maximises F1 on the **validation** split
  (the middle of the best range, for margin).
- *Augmentation:* the ML models also see 5 corrupted copies of the training split
  (noise from 5–40 dB, 0–20% dropped samples), so a noisy sensor isn't mistaken for a
  faulty machine.

**Severity (interpretable).** For each alert, every feature is compared with the healthy
profile (mean and std of healthy training windows). Severity comes from the largest deviation:

| Severity | Largest feature deviation | Roughly, for DE RMS on CWRU |
|---|---|---|
| low | < 10σ | up to ~+25% RMS |
| medium | 10σ – 100σ | up to ~3.5× healthy RMS |
| high | ≥ 100σ | more than ~3.5× healthy RMS |

The same deviations give the **top contributing features** shown with each alert,
e.g. `DE_rms (+272σ)` together with the value, healthy mean and "× healthy" ratio. The
bands are round numbers (decades) set by hand, not tuned on test data. Change
`SEVERITY_SIGMA` in `model.py`.

**Physics check (`physics.py`).** An independent check of each alert that uses no ML.
A damaged bearing produces a burst of vibration every time a ball passes the defect, at a rate
fixed by the bearing geometry and shaft speed. For the CWRU drive-end bearing (SKF 6205) at
1796 rpm these rates are: inner race 162 Hz, outer race 107 Hz, ball 141 Hz.
The check works in four steps:
1. take ~0.34 s of signal around the alert;
2. band-pass it and take its envelope;
3. look for a peak at each defect frequency and its 2nd harmonic;
4. compare the strongest peak with the ML prediction.

The verdict is **confirms**, **disagrees** or **inconclusive** (no peak ≥ 12 dB above the
background; healthy bearings sit at ~5–6 dB). The defect frequencies come from
`data_loader.py`: from each file's RPM for CWRU, or from `CSV_PHYSICS` for your own data.

**Maintenance decisions (`maintenance.py`).**
- *Recommended action:* each fault type has an action, e.g. "plan a bearing replacement;
  check the housing fit…". Each severity has an urgency: high = act within 24 h,
  medium = within 7 days, low = monitor.
- *Health index:* 100 = no alerts in the last 20 windows; each alert costs ⅓, ⅔ or 1 by severity.
- *Time to critical (prognostic):* a straight line fitted to the last 30 health values. While
  health is falling, it is extended to the critical level (25). CWRU recordings last seconds,
  so the estimates are in seconds; with run-to-failure data the same code estimates the
  remaining useful life in hours or days.

## Results on CWRU (held-out test split, 184 windows)

| | Precision | Recall | F1 | False-alarm rate |
|---|---|---|---|---|
| Baseline (RMS z-score) | 1.00 | 1.00 | 1.00 | 0.00 |
| Isolation Forest | 1.00 | 1.00 | 1.00 | 0.00 |
| Random Forest fault type (macro) | 1.00 | 1.00 | 1.00 | – |

**Be careful how you present this.** Each class here is a single ~5 s recording, so the
test windows come from the same bearing and the same run as the training windows. The
generalisation test below shows these perfect scores do **not** carry over to a new operating
point. Severity on clean test data: ball → medium, inner/outer race → high, normal → none.

Full tables and confusion matrices are in `reports/metrics.md`.

### Robustness (models trained on the training split; test split corrupted)

Detection F1. "No aug" = the same pipeline trained without augmentation, retrained inside `evaluate.py`.

| Condition | Baseline | IF (no aug) | IF (augmented) |
|---|---|---|---|
| clean | 1.00 | 1.00 | 1.00 |
| noise 30 dB | 1.00 | 1.00 | 1.00 |
| noise 20 dB | 1.00 | 0.86 | **1.00** |
| noise 10 dB | 1.00 | 0.86 | 0.86 |
| noise 0 dB | 0.99 | 0.86 | 0.86 |
| 5% dropped | 1.00 | 0.99 | 1.00 |
| 20% dropped (interpolated) | 1.00 | 0.86 | **1.00** |
| 20% dropped, zero-filled (naive) | 1.00 | 0.86 | 0.86 |

Noise SNR is measured against the healthy machine's signal (0 dB = noise as strong as the
healthy vibration). F1 = 0.86 means every healthy window was flagged (false-alarm rate
100%). What we learned:

- **The spectral features are what break.** A healthy bearing has almost no energy above
  12 kHz, so a little broadband noise, or the kinks left by naive zero-filling, makes those
  band shares jump by up to ~20×.
- **Augmentation plus interpolation fixes the moderate cases** (20 dB noise, 20% dropped).
  Clean performance is unchanged.
- **At 10 dB and below, noisy healthy windows look as unusual as the 0.007" ball fault**,
  and an unsupervised detector cannot tell which kind of "unusual" matters.
- **The simple RMS baseline is the most robust detector here**, because these faults raise
  RMS 2–17×. Keep it next to the model: if they disagree, look closer.
- **Severity has the same weakness.** Under heavy noise the high-frequency features deviate
  a lot, so false alarms can also be rated "high".

![Line charts of detection F1 against added noise (left) and dropped samples (right). The RMS baseline stays near 1.0; the Isolation Forest trained on clean data drops to 0.86 at 20 dB noise and 20% dropped samples; with augmentation it holds 1.0 until 10 dB noise](reports/robustness.png)

### Generalisation to unseen motor loads (the most honest result)

The 0 HP model is scored on every window of the 1–3 HP recordings, which have a different
load and shaft speed and were never used for training or tuning. `evaluate.py` downloads
these 12 extra files (~100 MB) on first run.

| Unseen load | IF detection F1 | IF false-alarm rate | Baseline F1 | Fault-type accuracy | End-to-end macro F1 |
|---|---|---|---|---|---|
| 1 HP | 0.88 | 85% | 1.00 | 76% | 0.56 |
| 2 HP | 0.87 | 89% | 1.00 | 70% | 0.48 |
| 3 HP | 0.90 | 69% | 1.00 | 68% | 0.52 |

- **The ML pipeline is tied to the operating point it was trained on.** At a new load, most
  healthy windows look abnormal to the Isolation Forest, and fault-type accuracy drops to ~70%.
- **The RMS baseline still separates healthy from faulty perfectly**, with no false alarms.
- **Next steps:** train on several loads, add speed-normalised features (order tracking or
  envelope peaks at defect frequencies), or let the baseline or physics check confirm
  Isolation Forest alerts.

These numbers were not used to tune anything.

### Physics check of the test alerts

| True fault | Physics confirms | Inconclusive | Disagrees |
|---|---|---|---|
| inner race | 46 | 0 | 0 |
| outer race | 46 | 0 | 0 |
| ball | 0 | 46 | 0 |

Inner- and outer-race alerts show clear peaks (around 25 dB) at their defect frequencies. The 0.007" ball
fault shows no clear peak at the ball frequency, which matches published analyses of this
dataset: ball defects don't strike the races consistently. The physics check never contradicts the ML.

## On hackathon day: swapping in their data

1. Look at their file. If the columns aren't `timestamp`, `label`, etc., edit the constants at
   the top of `data_loader.py`:
   - `CSV_TIMESTAMP_COL`, `CSV_LABEL_COL`: their column names;
   - `CSV_EXCLUDE_COLS`: numeric columns that are not sensors (IDs, counters);
   - `NORMAL_ALIASES`: whatever they call the healthy state;
   - `CSV_PHYSICS` (optional, vibration data from a rolling bearing only): the vibration
     column, shaft rpm and bearing defect orders. These turn on the physics check.

   Text columns are ignored automatically. Then check it loads:
   ```powershell
   python data_loader.py --source their_data.csv
   ```
2. Choose a window that covers a meaningful stretch of time, then train:
   ```powershell
   python model.py --data their_data.csv --window 60 --step 30 --bands 0   # e.g. 1 Hz SCADA data
   python evaluate.py
   streamlit run app.py
   ```
   The data source and every setting are stored in the model bundle, so `evaluate.py` and
   `app.py` pick them up automatically.

   **No-code alternative:** open the dashboard's **Data studio** page, upload the file, map the
   columns, and click *Train & activate*, then *Run evaluation*.

How the CSV loader behaves:
- **Sampling rate:** inferred from the timestamps (override with `--fs`).
- **Segments:** a new segment starts whenever the label changes or there is a time gap
  more than 5× the usual spacing.
- **Missing values:** interpolated.
- **No label column:** the pipeline runs in anomaly-only mode. It assumes the first 60% of
  the data is mostly healthy, fault types show as `unknown_anomaly`, and `evaluate.py`
  skips the metrics.

**Tips from testing on a synthetic slow multi-sensor CSV** (1 Hz temperature, pressure
and current; faults were a +15 °C shift and a −24% pressure drop):
- **Use `--bands 0` for slow data.** FFT bands on such data are mostly noise and dilute
  the Isolation Forest.
- **Even then the IF reached only F1 ≈ 0.85, while the RMS baseline was perfect.**
  Isolation Forest splits on randomly chosen features, so a large shift in 1–2 of ~18
  features gets diluted. Always compare against the baseline.
- **The window should span the time a fault takes to show up.** Training needs at least
  a few dozen windows per class.

## Accessibility

The dashboard is built to WCAG 2.1 AA, and `tests/test_accessibility.py` checks it
automatically (`python -m pytest tests`).

| WCAG criterion | How the dashboard meets it |
|---|---|
| 1.1.1 Non-text content | Every chart gets a generated text summary (e.g. "46 of 92 windows are above the alert threshold…") as its accessible description. **Show chart data as tables** also displays the summary and the data under each chart. Report images have descriptive alt text. |
| 1.3.1 Info and relationships | Real structure for assistive technology: one level-1 heading per page, level-2 section headings, labelled card groups, alert feeds as lists, machine tiles as articles. The health gauge is a `meter` ("Health 33 of 100, Degraded"). |
| 1.4.1 Use of colour | Severity is always written (HIGH / MEDIUM / LOW) and shown by marker shape in charts (circle, triangle, diamond, square); colour dots are decorative and hidden from screen readers. |
| 1.4.3 / 1.4.11 Contrast | All text is at least 4.5:1 (most is 10:1 or more); primary buttons use a darker blue so white text reaches 5.4:1; status colours are at least 3:1 against cards. |
| 1.4.4 Resize text | **Larger text** setting; sizes are in `rem`, so browser zoom works too. |
| 2.1.1 Keyboard | Every control is a native Streamlit widget, so the whole app works with Tab / Shift+Tab / Enter / arrow keys. |
| 2.2.1 / 2.2.2 Timing, pause | The live replay only starts on request, can be paused, and has **Step ▸** to advance one window at a time, at any pace. |
| 2.3.3 Animation | **Reduce motion** setting, and the OS "reduce motion" preference is respected automatically. |
| 2.4.1 Bypass blocks | "Skip to main content" link at the top of every page. |
| 2.4.7 Focus visible | 3 px high-contrast focus outline on every interactive element. |
| 3.3.2 Labels | Every input has a visible label and, where useful, help text. |

**Sidebar → ♿ Accessibility** offers four settings, applied to every page: *Larger text*,
*High contrast* (white on black, with outlines), *Reduce motion* (no animations; the replay
starts at half speed) and *Show chart data as tables*.

## Files

| File | Purpose |
|---|---|
| `data_loader.py` | Loads CWRU or a CSV into the standard format (the only format-aware file) |
| `features.py` | Chronological split, windowing, features, noise/drop simulation |
| `baseline.py` | RMS z-score detector and the F1-based threshold picker |
| `model.py` | Training, saving/loading, and `analyze()`: the shared inference path used by `evaluate.py` and the dashboard |
| `physics.py` | Envelope spectrum and the physics check of an alert against the bearing defect frequencies |
| `maintenance.py` | Recommended actions, health index, time-to-critical projection, incident grouping |
| `evaluate.py` | Test-split metrics, baseline vs model, robustness, physics check, unseen-load generalisation → `reports/` |
| `app.py` | Dashboard entry point: theme, sidebar, page navigation |
| `dashboard/ui.py`, `charts.py`, `state.py` | Shared look and feel, Altair charts, and cached data/model access, demo fleet, work-order storage |
| `dashboard/views/` | One module per page: `fleet`, `machine`, `work_orders`, `studio`, `lab` |
| `tests/test_accessibility.py` | Automated WCAG 2.1 AA checks for the dashboard |
| `reports/` | `metrics.md`, `metrics.json`, `confusion_matrices.png`, `robustness.png` |

Runtime files that are not in git: `models/` (rebuilt by `python model.py`), `data/raw/` (downloaded),
`data/work_orders.json` and `data/uploads/` (created by the dashboard).

## Dashboard

`streamlit run app.py` opens a dark-themed, multi-page app (theme in `.streamlit/config.toml`).
The sidebar holds the page menu and the **Simulate sensor problems** controls (added noise,
dropped samples), which apply to every page.

| Page | What it shows |
|---|---|
| 🏭 **Fleet overview** (home) | Counts of healthy, at-risk and critical machines plus open work orders; a tile per machine with health gauge, state, latest fault, time to critical and a health sparkline (**Open machine →** drills in); the health of every machine over time; a risk ranking; the latest alerts across the fleet |
| ⚙️ **Machine detail** | For the selected machine: health, latest status, time to critical and alert counts, then four tabs. **Overview**: anomaly-score timeline, health trend with projection, alerts by fault type and severity, alert feed. **Live replay**: plays the data back window by window with pop-up notifications for new alerts. **Alerts & evidence**: alert table with CSV report and, per alert, recommended action, physics check, waveform, spectrum, envelope spectrum and top contributing features. **Incidents**: alerts grouped into incidents with *Create work order* |
| 🛠 **Work orders** | Open / in progress / overdue / done counts; an editable table (status, assignee, notes) saved to `data/work_orders.json`; due dates by urgency (high 24 h, medium 7 days, low 30 days); bulk creation for all incidents; CSV export |
| 📤 **Data studio** | Upload a CSV (or use the built-in sample), map the timestamp, label and sensor columns, check the dataset, then **analyse it with the active model** or **train a new model** (window, step, FFT bands, augmentation) and run the evaluation, all without code. **Restore the CWRU demo model** puts the original back |
| 🧪 **Model lab** | Model card; **threshold playground** (precision, recall, F1 and false-alarm rate as you move the alert threshold, on the validation split); Random Forest feature importance; the full held-out evaluation (confusion matrices, robustness, unseen-load generalisation, physics check) |

**The demo fleet** is built from the held-out test recordings, which are never used for
training or tuning. Motor M-101 stays healthy. M-102, M-103 and M-104 run healthy and then
develop a ball, inner-race and outer-race fault respectively. With your own labelled data,
the fleet is built the same way from its test split.

**Suggested demo:** start on the Fleet overview, then click **Open machine →** on M-103 and
play **Live replay**. Move the noise slider to show robustness, raise a work order from
**Incidents** and show it on **Work orders**. Load the sample in the **Data studio**, and end
with the threshold slider in the **Model lab**.
