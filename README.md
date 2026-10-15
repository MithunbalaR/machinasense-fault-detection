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
| 4. Dashboard | `streamlit run app.py` | Alert list with severity, fault type, top features and the signal window |

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

## Results on CWRU (held-out test split, 184 windows)

| | Precision | Recall | F1 | False-alarm rate |
|---|---|---|---|---|
| Baseline (RMS z-score) | 1.00 | 1.00 | 1.00 | 0.00 |
| Isolation Forest | 1.00 | 1.00 | 1.00 | 0.00 |
| Random Forest fault type (macro) | 1.00 | 1.00 | 1.00 | – |

**Be careful how you present this.** Each class here is a single ~5 s recording, so the
test windows come from the same bearing and the same run as the training windows. Perfect
scores on CWRU are well known and do not mean the model generalises to new machines.
Severity on clean test data: ball → medium, inner/outer race → high, normal → none.

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

![robustness](reports/robustness.png)

## On hackathon day: swapping in their data

1. Look at their file. If the columns aren't `timestamp`, `label`, etc., edit the constants at
   the top of `data_loader.py`:
   - `CSV_TIMESTAMP_COL`, `CSV_LABEL_COL`: their column names;
   - `CSV_EXCLUDE_COLS`: numeric columns that are not sensors (IDs, counters);
   - `NORMAL_ALIASES`: whatever they call the healthy state.

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

## Files

| File | Purpose |
|---|---|
| `data_loader.py` | Loads CWRU or a CSV into the standard format (the only format-aware file) |
| `features.py` | Chronological split, windowing, features, noise/drop simulation |
| `baseline.py` | RMS z-score detector and the F1-based threshold picker |
| `model.py` | Training, saving/loading, and `analyze()`: the shared inference path used by `evaluate.py` and `app.py` |
| `evaluate.py` | Test-split metrics, baseline vs model, robustness test → `reports/` |
| `app.py` | Streamlit dashboard |
| `reports/` | `metrics.md`, `metrics.json`, `confusion_matrices.png`, `robustness.png` |

## Dashboard

`streamlit run app.py` opens a dark-themed dashboard (theme in `.streamlit/config.toml`).

**Header row** (always visible):
- **Machine health:** a 0–100 gauge, where 100 = no alerts in the last 20 windows; each alert
  costs ⅓, ⅔ or 1 by severity. 80+ is *Healthy*, 50+ *Watch*, 25+ *Degraded*, below that *Critical*.
- **Latest status:** the fault and severity of the most recent window.
- **Counts:** windows analysed, alerts by severity, and alerts per fault type.

**Tabs:**

| Tab | What it shows |
|---|---|
| Overview | Anomaly-score timeline coloured by severity with the threshold line; alerts by fault type and by severity; a feed of the latest alerts |
| Live replay | Plays the data back window by window as if it were streaming from the machine. Play / pause / restart and a speed control; the health gauge, status, score chart, current waveform and alert feed update live |
| Alerts & evidence | Filterable alert table. For the selected alert: severity, fault, window time and score; then a waveform **and** frequency spectrum per sensor against a healthy reference, and the top contributing features on a log σ scale with the medium/high severity bands marked |
| Evaluation | Headline F1 scores, baseline vs model table, confusion matrices and robustness charts, read from `reports/metrics.json` (run `python evaluate.py` first) |

The sidebar switches between the test, validation and whole dataset, and can inject noise or
dropped samples live. Combined with **Live replay**, that's a good way to demo robustness to judges.
