from __future__ import annotations

import pandas as pd


def subtract_background(flux_series: pd.Series, window_minutes: int = 60) -> pd.Series:
    """
    Subtract quiet-Sun background using rolling minimum.
    Physics: quiet-Sun background = median of preceding 60-min low-activity window.
    Ref: standard practice in solar X-ray analysis (Benz 2017, Living Reviews)
    """
    flux = pd.to_numeric(flux_series, errors="coerce").ffill().bfill().fillna(0)
    background = flux.rolling(window=window_minutes, min_periods=1).quantile(0.1)
    return (flux - background).clip(lower=0)


def savitzky_golay_derivative(
    flux_series: pd.Series, window: int = 11, polyorder: int = 3
) -> pd.Series:
    """
    Compute temporal derivative using Savitzky-Golay filter.
    Ref: Savitzky & Golay (1964), Analytical Chemistry 36, 1627
    Window=11 samples @ 1-min cadence = 11-min smoothing window.
    """
    from scipy.signal import savgol_filter

    flux = pd.to_numeric(flux_series, errors="coerce").ffill().bfill().fillna(0)
    if len(flux) < 3:
        return flux.diff().fillna(0)
    safe_window = min(window, len(flux) if len(flux) % 2 == 1 else len(flux) - 1)
    safe_window = max(safe_window, polyorder + 2)
    if safe_window % 2 == 0:
        safe_window += 1
    if safe_window > len(flux):
        return flux.diff().fillna(0)
    smoothed = savgol_filter(
        flux.to_numpy(),
        window_length=safe_window,
        polyorder=min(polyorder, safe_window - 2),
        deriv=1,
    )
    return pd.Series(smoothed, index=flux.index)
