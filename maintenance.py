"""From alerts to maintenance decisions: actions, machine health and time to critical.

Health index: 100 = no alerts in the last RECENT windows. Each alert costs 1/3, 2/3
or 1 (low / medium / high severity), averaged over the recent windows.

Time to critical: a straight line is fitted to the health index over the last
TREND_WINDOWS windows. If health is falling, the line is extended to the
CRITICAL level. This is a simple, explainable prognostic. With run-to-failure data
(hours or days of a machine degrading), the same projection gives a remaining
useful life estimate.
"""
import numpy as np
import pandas as pd

SEVERITY_WEIGHT = {"none": 0.0, "low": 1 / 3, "medium": 2 / 3, "high": 1.0}
RECENT = 20             # windows behind the health index
TREND_WINDOWS = 30      # windows used to fit the health trend
CRITICAL = 25           # health below this is "Critical"
HEALTH_STATES = [(80, "good", "Healthy"), (50, "warning", "Watch"), (CRITICAL, "serious", "Degraded"),
                 (-1, "critical", "Critical")]

ACTIONS = {
    "inner_race": "Inner race defect. Plan a bearing replacement; check the shaft fit and lubrication.",
    "outer_race": "Outer race defect. Plan a bearing replacement; check the housing fit, alignment and lubrication.",
    "ball": "Rolling element (ball) defect. Plan a bearing replacement; check the lubricant for contamination.",
}
DEFAULT_ACTION = "Abnormal behaviour. Inspect the machine and review recent maintenance or process changes."
URGENCY = {
    "high": "Act within 24 h: reduce load or stop the machine if possible.",
    "medium": "Schedule maintenance within 7 days and increase monitoring.",
    "low": "Monitor closely; check at the next planned maintenance.",
}


SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
INCIDENT_COLUMNS = ["start_window", "end_window", "start_time", "end_time", "fault_type", "severity", "windows", "peak_sigma"]


def incidents(results, max_gap=2):
    """Group alert windows into incidents: runs of alerts with the same predicted fault,
    allowing gaps of up to `max_gap` quiet windows. One incident -> one work order."""
    rows, current = [], None
    for idx, r in results[results.is_alert].iterrows():
        if current and r.fault_type == current["fault_type"] and idx - current["end_window"] <= max_gap + 1:
            current.update(end_window=idx, end_time=r.end_time, windows=current["windows"] + 1,
                           peak_sigma=max(current["peak_sigma"], r.max_deviation_sigma))
            if SEVERITY_RANK[r.severity] > SEVERITY_RANK[current["severity"]]:
                current["severity"] = r.severity
        else:
            if current:
                rows.append(current)
            current = {"start_window": idx, "end_window": idx, "start_time": r.timestamp, "end_time": r.end_time,
                       "fault_type": r.fault_type, "severity": r.severity, "windows": 1, "peak_sigma": r.max_deviation_sigma}
    if current:
        rows.append(current)
    return pd.DataFrame(rows, columns=INCIDENT_COLUMNS)


def recommend(fault_type, severity):
    return {"action": ACTIONS.get(fault_type, DEFAULT_ACTION), "urgency": URGENCY.get(severity, "")}


def health_index(severities):
    """Rolling health index (0-100) for each window, from the alert severities."""
    load = pd.Series(severities).map(SEVERITY_WEIGHT).rolling(RECENT, min_periods=1).mean()
    return 100 * (1 - load.to_numpy())


def health_state(score):
    """(status colour key, label) for a health score."""
    return next((key, label) for limit, key, label in HEALTH_STATES if score >= limit)


def time_to_critical(health, seconds_per_window):
    """Project the recent health trend to the critical level.

    Returns None when there is not enough history, health is already critical,
    or the trend is not falling; otherwise a dict with the slope and the
    estimated number of windows and seconds until health reaches CRITICAL.
    """
    recent = np.asarray(health[-TREND_WINDOWS:], float)
    if len(recent) < 5 or recent[-1] < CRITICAL:
        return None
    slope, intercept = np.polyfit(np.arange(len(recent)), recent, 1)   # health points per window
    if slope > -0.1:
        return None
    now = intercept + slope * (len(recent) - 1)
    windows = max((CRITICAL - now) / slope, 0)
    return {"slope_per_window": slope, "windows": windows, "seconds": windows * seconds_per_window}


def format_duration(seconds):
    for unit, size in [("days", 86400), ("h", 3600), ("min", 60), ("s", 1)]:
        if seconds >= size:
            return f"{seconds / size:.1f} {unit}"
    return f"{seconds * 1000:.0f} ms"
