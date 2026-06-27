from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .fai import C1_FLUX, compute_fai
from .ladder import InstabilityLadderState, assign_ladder_state, replay_ladder
from .nri import nri_sigma_series


def goes_class_from_flux(peak_flux: float) -> str:
    if peak_flux >= 1e-4:
        return f"X{peak_flux / 1e-4:.1f}"
    if peak_flux >= 1e-5:
        return f"M{peak_flux / 1e-5:.1f}"
    if peak_flux >= 1e-6:
        return f"C{peak_flux / 1e-6:.1f}"
    if peak_flux >= 1e-7:
        return f"B{peak_flux / 1e-7:.1f}"
    return f"A{peak_flux / 1e-8:.1f}"


def detect_flare_events(
    sxr_series: pd.Series,
    nri_series: pd.Series,
    sigma_nri: float | None = None,
    fai_series: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Automated flare catalogue builder using dual-trigger logic.

    Citation:
    - GOES class boundaries: Thomas et al. (1985), Solar Physics 95, 323
    - Dual-trigger logic: AgniDrishti proposal, Team HelioDynamics, BAH 2026
    - Coincidence window: Veronig et al. (2005), A&A 431, 1047
    """
    sxr = pd.to_numeric(sxr_series, errors="coerce").ffill().bfill().fillna(0)
    nri = pd.to_numeric(nri_series, errors="coerce").reindex(sxr.index).interpolate().ffill().bfill().fillna(0)
    nri_sigma = nri_sigma_series(nri, sxr, sigma_nri)
    fai = compute_fai(sxr) if fai_series is None else fai_series.reindex(sxr.index).interpolate().fillna(0)
    readings = replay_ladder(fai, nri_sigma)
    ladder_names = pd.Series([reading.state.name for reading in readings], index=sxr.index)

    soft = (sxr >= C1_FLUX) & (sxr.diff().fillna(0) > 0)
    hard = nri_sigma > 3
    hard_window = hard.rolling(window=5, center=True, min_periods=1).max().astype(bool)
    trigger = soft & hard_window

    events = []
    trigger_times = list(trigger[trigger].index)
    groups: list[list[pd.Timestamp]] = []
    for timestamp in trigger_times:
        if not groups or (timestamp - groups[-1][-1]) > pd.Timedelta(minutes=30):
            groups.append([timestamp])
        else:
            groups[-1].append(timestamp)

    for group in groups:
        group_start = group[0]
        start_position = sxr.index.get_loc(group_start)
        while start_position > 0 and sxr.iloc[start_position - 1] >= C1_FLUX:
            start_position -= 1
        start_time = sxr.index[start_position]
        search_end = min(group_start + pd.Timedelta(minutes=150), sxr.index[-1])
        peak_slice = sxr.loc[start_time:search_end]
        if peak_slice.empty:
            continue
        peak_time = peak_slice.idxmax()
        peak_flux = float(peak_slice.max())
        end_time = sxr.index[-1]
        half_peak = max(C1_FLUX, peak_flux * 0.5)
        for candidate_time, value in sxr.loc[peak_time:].items():
            if candidate_time > peak_time and value <= half_peak:
                end_time = candidate_time
                break
        event_slice = sxr.loc[start_time:end_time]
        event_nri_sigma = nri_sigma.loc[event_slice.index]
        event_nri_slope = event_nri_sigma.diff().fillna(0)
        event_ladder_states = pd.Series(
            [
                assign_ladder_state(float(fai.loc[timestamp]), float(event_nri_sigma.loc[timestamp]), float(event_nri_slope.loc[timestamp])).name
                for timestamp in event_slice.index
            ],
            index=event_slice.index,
        )
        max_ladder_state = max(
            (InstabilityLadderState[state_name] for state_name in event_ladder_states),
            key=lambda state: state.value,
        )
        critical_times = event_ladder_states[event_ladder_states.isin(["RED", "CRITICAL"])]
        first_critical_time = critical_times.index[0] if len(critical_times) else group_start
        lead_time = (
            int((peak_time - first_critical_time).total_seconds() / 60)
            if first_critical_time <= peak_time
            else 0
        )
        events.append(
            {
                "start_time": start_time,
                "peak_time": peak_time,
                "end_time": end_time,
                "first_critical_time": first_critical_time,
                "goes_class": goes_class_from_flux(peak_flux),
                "peak_flux": peak_flux,
                "peak_fai": float(fai.loc[event_slice.index].max()),
                "peak_nri": float(nri_sigma.loc[event_slice.index].max()),
                "ladder_state": max_ladder_state.name,
                "lead_time_min": lead_time,
            }
        )
    return pd.DataFrame(events)


def save_events_sqlite(events: pd.DataFrame, path: str | Path = "catalogue/flare_catalogue.sqlite") -> None:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        events.to_sql("events", conn, if_exists="replace", index_label="id")


def load_events_sqlite(path: str | Path = "catalogue/flare_catalogue.sqlite") -> pd.DataFrame:
    db_path = Path(path)
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM events", conn)


def build_event_labels(index: pd.DatetimeIndex, events: pd.DataFrame, horizon_minutes: int = 30) -> np.ndarray:
    labels = np.zeros(len(index), dtype=int)
    if events.empty:
        return labels
    for peak in pd.to_datetime(events["peak_time"]):
        start = peak - pd.Timedelta(minutes=horizon_minutes)
        labels[(index >= start) & (index <= peak)] = 1
    return labels
