"""independent_characterization.py

Physics-informed independent characterization for SoLEXS (soft X-ray)
and HEL1OS (hard X-ray) detectors on Aditya-L1, plus cross-instrument
fusion via a Neupert-effect-aware time tolerance.

Design:
  - characterize_solexs  : thermal flare detector on SDD2 count-rate
  - characterize_hel1os  : impulsive burst detector on CZT fullband
  - fuse_independent_detections : Neupert-aware temporal matching

Detection thresholds
--------------------
SoLEXS  : quiet-Sun background estimated from the low-20th-percentile
           of the full-day SDD2 count-rate.  A flare trigger requires
           the smoothed count-rate to exceed bg + 3.5 σ for ≥ 2 min.
           (Lowered from any implicit hard cut to catch C-/M-class
           events even on high-background days.)

HEL1OS  : CZT background is almost entirely zeros between photon
           arrivals at 1-s cadence.  We threshold on the 85th percentile
           of the *non-zero* sample distribution (≈ "bright" photon
           bursts).  Events require ≥ 30 s of sustained elevation.
           If the non-zero population is too small (< 20 samples) the
           threshold falls back to the 99th-percentile of all samples
           so sparse coverage days do not generate phantom detections.

Fusion tolerance
----------------
45 minutes.  The Neupert effect (Neupert 1968, ApJ 153, L59) predicts
that the HXR peak (non-thermal electrons) precedes or coincides with
the SXR peak (thermal plasma heated by those electrons); the gap
depends on chromospheric evaporation timescales and is typically
10–30 min but can reach 40 min for long-duration events
(Veronig et al. 2005, A&A 431, 1047).  We use 45 min to safely
accommodate the full observed distribution.

Partial-coverage handling
-------------------------
When HEL1OS data does not cover the time of a SoLEXS flare (e.g.
the satellite was in the Earth's shadow or data are unavailable for
that orbit), the SoLEXS event is still included in the fused catalogue
with ``match_type = "SoLEXS-only (no HEL1OS coverage)"`` rather than
being silently dropped.  This preserves the full thermal event record
and makes coverage gaps explicit.

Citations
---------
- Neupert (1968), ApJ 153, L59
- Veronig et al. (2005), A&A 431, 1047
- Sarwade et al. (2025), arXiv:2509.26292 (SoLEXS instrument)
- Nandi et al. (2025), arXiv:2512.12679 (HEL1OS instrument)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

_MERGE_GAP_SXR_S = 1800      # 30 min: merge SoLEXS intervals closer than this
_MERGE_GAP_HXR_S = 900       # 15 min: merge HEL1OS intervals closer than this
_MIN_DURATION_SXR_S = 120    # 2 min minimum SoLEXS event duration
_MIN_DURATION_HXR_S = 30     # 30 s  minimum HEL1OS burst duration
_SIGMA_MULT_SXR = 3.5        # sigma multiplier for SoLEXS threshold
_QUIET_PERCENTILE = 0.20     # bottom fraction used to estimate quiet-Sun background


def _build_contiguous_events(
    series: pd.Series,
    threshold: float,
    min_duration_s: float,
    merge_gap_s: float,
) -> list[dict]:
    """Return list of {start, peak, end, peak_val} dicts for intervals above *threshold*."""
    above = (series >= threshold).astype(int)
    trans = above.diff().fillna(0)
    starts_idx = series.index[trans == 1]
    ends_idx = series.index[trans == -1]

    raw: list[dict] = []
    for start in starts_idx:
        matching = [e for e in ends_idx if e > start]
        end = matching[0] if matching else series.index[-1]
        duration_s = (end - start).total_seconds()
        if duration_s < min_duration_s:
            continue
        window = series.loc[start:end]
        raw.append(
            {
                "start": start,
                "peak": window.idxmax(),
                "end": end,
                "peak_val": float(window.max()),
                "duration_s": duration_s,
            }
        )

    # Merge consecutive events closer than merge_gap_s
    merged: list[dict] = []
    for ev in raw:
        if merged and (ev["start"] - merged[-1]["end"]).total_seconds() < merge_gap_s:
            if ev["peak_val"] > merged[-1]["peak_val"]:
                merged[-1]["peak"] = ev["peak"]
                merged[-1]["peak_val"] = ev["peak_val"]
            merged[-1]["end"] = ev["end"]
            merged[-1]["duration_s"] = (merged[-1]["end"] - merged[-1]["start"]).total_seconds()
        else:
            merged.append(dict(ev))
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def characterize_solexs(
    sdd1: pd.DataFrame,
    sdd2: pd.DataFrame,
    sdd1_gti: pd.DataFrame,
    sdd2_gti: pd.DataFrame,
) -> pd.DataFrame:
    """Detect and characterise thermal flares from SoLEXS SDD2 count-rate.

    Uses SDD2 as the primary channel (SDD1 is optional; used only if SDD2 is
    unavailable).  GTI tables are accepted but not currently used to mask data
    (the SDD2 count-rate already encodes valid intervals via NaN-free segments).

    Returns a DataFrame with one row per detected flare:
      start, peak_time, end, peak_cts, rise_time_s, decay_tau_s, sigma_above_bg
    """
    # Select primary channel: prefer SDD2
    if sdd2 is not None and not sdd2.empty and "counts" in sdd2.columns:
        raw = sdd2["counts"].dropna()
    elif sdd1 is not None and not sdd1.empty and "counts" in sdd1.columns:
        raw = sdd1["counts"].dropna()
    else:
        return pd.DataFrame()

    if len(raw) < 60:
        return pd.DataFrame()

    # Smooth at 10-sample scale to suppress single-sample spikes.
    # Use a rolling median rather than resample() to avoid the costly
    # re-indexing of a 86 400-row 1-s series to a new DatetimeIndex.
    sxr = raw.rolling(10, min_periods=1, center=True).median().ffill().bfill()

    # Quiet-Sun background from the lowest-20th-percentile samples
    quiet_vals = sxr[sxr <= sxr.quantile(_QUIET_PERCENTILE)].dropna()
    if quiet_vals.empty:
        quiet_vals = sxr.dropna().iloc[: max(30, int(len(sxr) * 0.2))]
    bg = float(quiet_vals.median())
    sigma = float(quiet_vals.std(ddof=0)) if len(quiet_vals) > 2 else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(sxr.diff().abs().quantile(0.95)) or 1.0
    threshold = bg + _SIGMA_MULT_SXR * sigma

    events = _build_contiguous_events(sxr, threshold, _MIN_DURATION_SXR_S, _MERGE_GAP_SXR_S)

    rows: list[dict] = []
    for ev in events:
        start, peak_time, end, peak_cts = ev["start"], ev["peak"], ev["end"], ev["peak_val"]
        rise_time_s = max((peak_time - start).total_seconds(), 1.0)
        # Estimate e-folding decay time from half-peak to end
        half_peak = max(bg, peak_cts * 0.5)
        decay_tau_s = np.nan
        decay_window = sxr.loc[peak_time:end]
        below_half = decay_window[decay_window <= half_peak]
        if not below_half.empty:
            decay_tau_s = float((below_half.index[0] - peak_time).total_seconds())
        rows.append(
            {
                "start": start,
                "peak_time": peak_time,
                "end": end,
                "peak_cts": round(peak_cts, 2),
                "rise_time_s": round(rise_time_s, 1),
                "decay_tau_s": round(float(decay_tau_s), 1) if np.isfinite(float(decay_tau_s) if not np.isnan(decay_tau_s) else float("nan")) else np.nan,
                "sigma_above_bg": round((peak_cts - bg) / max(sigma, 1e-9), 2),
            }
        )
    return pd.DataFrame(rows)


def characterize_hel1os(
    cdte: pd.DataFrame,
    czt: pd.DataFrame,
    cdte_gti: pd.DataFrame,
    czt_gti: pd.DataFrame,
) -> pd.DataFrame:
    """Detect impulsive hard X-ray bursts from HEL1OS CZT fullband count-rate.

    Uses the CZT full-band (18–160 keV) proxy column produced by
    ``load_aditya_l1_dataset``.  Falls back to CdTe if CZT is unavailable.

    Returns a DataFrame with one row per detected burst:
      start, peak_time, end, peak_ctr, burst_duration_s, sigma_above_bg
    """
    # Select primary channel: CZT fullband
    hxr: pd.Series | None = None
    for df, col in [(czt, "czt_fullband_ctr"), (cdte, "cdte_fullband_ctr")]:
        if df is not None and not df.empty and col in df.columns:
            candidate = df[col].dropna()
            if len(candidate) >= 30:
                hxr = candidate
                break

    if hxr is None or len(hxr) < 30:
        return pd.DataFrame()

    # Background: CZT samples are sparse (most bins = 0).
    # Use non-zero samples for statistics; fall back to all-samples quantile.
    nonzero = hxr[hxr > 0]
    if len(nonzero) >= 20:
        # Use 85th percentile of non-zero population as burst threshold.
        # This avoids the degenerate case where std=0 (all background=0).
        burst_threshold = float(nonzero.quantile(0.85))
        bg_hxr = float(hxr.quantile(0.5))   # median (likely 0 on quiet days)
        sigma_hxr = float(nonzero.std(ddof=0)) if len(nonzero) > 2 else 1.0
    else:
        # Very sparse data: use 99th percentile of all samples
        burst_threshold = float(hxr.quantile(0.99))
        bg_hxr = 0.0
        sigma_hxr = 1.0

    # Ensure threshold is positive
    if not np.isfinite(burst_threshold) or burst_threshold <= 0:
        return pd.DataFrame()

    events = _build_contiguous_events(hxr, burst_threshold, _MIN_DURATION_HXR_S, _MERGE_GAP_HXR_S)

    rows: list[dict] = []
    for ev in events:
        start, peak_time, end, peak_ctr = ev["start"], ev["peak"], ev["end"], ev["peak_val"]
        burst_duration_s = (end - start).total_seconds()
        rows.append(
            {
                "start": start,
                "peak_time": peak_time,
                "end": end,
                "peak_ctr": round(peak_ctr, 2),
                "burst_duration_s": round(burst_duration_s, 1),
                "sigma_above_bg": round((peak_ctr - bg_hxr) / max(sigma_hxr, 1e-9), 2),
            }
        )
    return pd.DataFrame(rows)


# Neupert-effect fusion tolerance: 45 minutes
# The hard X-ray impulsive phase (HEL1OS) precedes the soft X-ray thermal
# peak (SoLEXS) by typically 10–30 min.  We use 45 min to cover the full
# observed distribution and long-duration events (Veronig et al. 2005).
_FUSION_TOLERANCE_MIN = 45


def fuse_independent_detections(
    solexs_events: pd.DataFrame,
    hel1os_events: pd.DataFrame,
    tolerance_minutes: float = _FUSION_TOLERANCE_MIN,
) -> pd.DataFrame:
    """Fuse SoLEXS thermal events with HEL1OS impulsive bursts.

    Matching logic
    --------------
    For each SoLEXS event, we search for any HEL1OS burst whose **peak_time**
    falls within ±``tolerance_minutes`` of the SoLEXS **start** time (or its
    peak — whichever gives the smallest gap).  This is more physically correct
    than comparing both peaks, because the Neupert effect predicts the HXR
    peak to *precede* the SXR peak.

    Partial-coverage handling
    -------------------------
    If there are no HEL1OS events at all (e.g. the satellite was not observing
    during the flare, or the data file does not exist), SoLEXS events are
    still returned as ``match_type = "SoLEXS-only (no HEL1OS coverage)"``.
    If HEL1OS data exist but no match is found, a SoLEXS event is tagged
    ``match_type = "SoLEXS-only (no HEL1OS match in window)"``.

    Returns
    -------
    pd.DataFrame with columns:
      solexs_start, solexs_peak, solexs_end, solexs_peak_cts,
      hel1os_peak (NaT if no match), hel1os_peak_ctr (NaN if no match),
      neupert_delay_min, match_type
    """
    if solexs_events is None or solexs_events.empty:
        return pd.DataFrame()

    no_hel1os = hel1os_events is None or hel1os_events.empty
    tol = pd.Timedelta(minutes=tolerance_minutes)

    rows: list[dict] = []
    for _, sev in solexs_events.iterrows():
        s_start = pd.Timestamp(sev["start"])
        s_peak = pd.Timestamp(sev["peak_time"])

        best_match: dict | None = None
        best_gap_min: float = float("inf")

        if not no_hel1os:
            for _, hev in hel1os_events.iterrows():
                h_peak = pd.Timestamp(hev["peak_time"])
                # Gap: HXR peak relative to SXR *start* (Neupert: HXR leads SXR)
                gap_to_start = (s_start - h_peak).total_seconds() / 60
                # Also check peak-to-peak gap for cases where SXR rise is fast
                gap_to_peak = abs((s_peak - h_peak).total_seconds()) / 60
                # Accept if HXR peak is within tolerance of SXR start or SXR peak
                # (HXR should precede SXR start by up to tolerance, or lag by up to 10 min)
                h_before_s_start = h_peak <= s_start + pd.Timedelta(minutes=10)
                within_window = (
                    (abs(gap_to_start) <= tolerance_minutes and h_before_s_start)
                    or gap_to_peak <= tolerance_minutes
                )
                if within_window:
                    effective_gap = min(abs(gap_to_start), gap_to_peak)
                    if effective_gap < best_gap_min:
                        best_gap_min = effective_gap
                        best_match = hev.to_dict()

        if best_match is not None:
            h_peak_ts = pd.Timestamp(best_match["peak_time"])
            neupert_delay = (s_peak - h_peak_ts).total_seconds() / 60
            rows.append(
                {
                    "solexs_start": s_start,
                    "solexs_peak": s_peak,
                    "solexs_end": pd.Timestamp(sev["end"]),
                    "solexs_peak_cts": sev.get("peak_cts", np.nan),
                    "hel1os_peak": h_peak_ts,
                    "hel1os_peak_ctr": best_match.get("peak_ctr", np.nan),
                    "neupert_delay_min": round(neupert_delay, 1),
                    "match_type": "Fused (SoLEXS + HEL1OS)",
                }
            )
        else:
            match_reason = (
                "SoLEXS-only (no HEL1OS coverage)"
                if no_hel1os
                else "SoLEXS-only (no HEL1OS match in window)"
            )
            rows.append(
                {
                    "solexs_start": s_start,
                    "solexs_peak": s_peak,
                    "solexs_end": pd.Timestamp(sev["end"]),
                    "solexs_peak_cts": sev.get("peak_cts", np.nan),
                    "hel1os_peak": pd.NaT,
                    "hel1os_peak_ctr": np.nan,
                    "neupert_delay_min": np.nan,
                    "match_type": match_reason,
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Format timestamps as HH:MM UTC strings for display legibility
    for col in ["solexs_start", "solexs_peak", "solexs_end"]:
        df[col] = pd.to_datetime(df[col])
    return df
