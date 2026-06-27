from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_metrics(predictions, labels, lead_times):
    """
    Returns dict with TPR, FAR, false alerts per day, TSS, HSS, AUC-ROC,
    median lead time with IQR, and CSI.

    Benchmark references embedded:
    - TSS=0.74, AUC=0.87: Hassani et al. (2025), ApJS 279, 27
    - HSS benchmark: Bloomfield et al. (2012), ApJ 747, 41
    - FAR < 0.30 target: Quantum-Ark Helios-Cortex comparison
    - Lead time > 10 min: BAH 2026 problem statement requirement
    """
    y_pred = np.asarray(predictions).astype(int)
    y_true = np.asarray(labels).astype(int)
    leads = np.asarray(lead_times, dtype=float)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    pod = tp / (tp + fn) if tp + fn else 0.0
    far = fp / (tp + fp) if tp + fp else 0.0
    pofd = fp / (fp + tn) if fp + tn else 0.0
    tss = pod - pofd
    hss_den = 2 * ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    hss = (2 * (tp * tn - fp * fn) / hss_den) if hss_den else 0.0
    csi = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    valid_leads = leads[np.isfinite(leads) & (leads >= 0)]
    auc = 0.0
    if len(np.unique(y_true)) == 2:
        try:
            auc = float(roc_auc_score(y_true, y_pred))
        except ValueError:
            auc = 0.0
    return {
        "TPR": float(pod),
        "POD": float(pod),
        "FAR": float(far),
        "false_alerts_per_day": float(fp),
        "TSS": float(tss),
        "HSS": float(hss),
        "AUC_ROC": float(auc),
        "median_lead_time": float(np.median(valid_leads)) if len(valid_leads) else 0.0,
        "lead_time_iqr": (
            float(np.percentile(valid_leads, 25)) if len(valid_leads) else 0.0,
            float(np.percentile(valid_leads, 75)) if len(valid_leads) else 0.0,
        ),
        "CSI": float(csi),
        "confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
    }


def compute_metrics_by_class(predictions, labels, class_series, events) -> dict[str, dict[str, float]]:
    y_pred = pd.Series(np.asarray(predictions).astype(int))
    y_true = pd.Series(np.asarray(labels).astype(int))
    classes = pd.Series(class_series).reset_index(drop=True).fillna("No event").astype(str)
    events_df = events if isinstance(events, pd.DataFrame) else pd.DataFrame(events)

    if "goes_class" in events_df:
        ordered_classes = []
        for value in events_df["goes_class"].dropna().astype(str):
            key = value[:1].upper()
            if key not in ordered_classes:
                ordered_classes.append(key)
    else:
        ordered_classes = []

    for fallback in ["B", "C", "M", "X"]:
        if fallback not in ordered_classes:
            ordered_classes.append(fallback)

    results: dict[str, dict[str, float]] = {}
    for event_class in ordered_classes:
        mask = classes.str.upper().str.startswith(event_class)
        subset_pred = y_pred[mask]
        subset_true = y_true[mask]
        event_mask = (
            events_df.get("goes_class", pd.Series(dtype=str))
            .astype(str)
            .str.upper()
            .str.startswith(event_class)
            if not events_df.empty
            else pd.Series(dtype=bool)
        )
        leads = (
            pd.to_numeric(events_df.loc[event_mask, "lead_time_min"], errors="coerce").to_numpy(dtype=float)
            if "lead_time_min" in events_df
            else np.array([], dtype=float)
        )

        if len(subset_true) == 0:
            results[event_class] = {"TPR": 0.0, "FAR": 0.0, "median_lead_time": 0.0, "N_events": int(event_mask.sum())}
            continue

        metrics = compute_metrics(subset_pred, subset_true, leads)
        results[event_class] = {
            "TPR": float(metrics["TPR"]),
            "FAR": float(metrics["FAR"]),
            "median_lead_time": float(metrics["median_lead_time"]),
            "N_events": int(event_mask.sum()),
        }
    return results
