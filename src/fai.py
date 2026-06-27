from __future__ import annotations

import numpy as np
import pandas as pd


C1_FLUX = 1e-6


def _minutes_since_c1(sxr_flux: pd.Series) -> pd.Series:
    above = sxr_flux >= C1_FLUX
    values: list[float] = []
    counter = 0
    for is_above in above:
        if is_above:
            counter += 1
            values.append(counter)
        else:
            counter = 0
            values.append(0)
    return pd.Series(values, index=sxr_flux.index, dtype=float)


def compute_fai(sxr_flux: pd.Series, cadence_minutes: int = 1) -> pd.Series:
    """
    Flare Anticipation Index -- thermal preconditioning signal from SoLEXS.
    Uses GOES XRS-B as SoLEXS proxy.

    Physics: FAI = 0.3*g + 0.3*p + 0.2*d + 0.2*s

    Citations:
    - FAI formula: AgniDrishti proposal, Team HelioDynamics, BAH 2026
    - EM-flux scaling: White et al. (2005), Solar Physics 227, 231
    - Thermal preconditioning concept: Benz (2017), Living Reviews in Solar Physics
    """
    # Data: GOES XRS proxy for SoLEXS (Lemen et al. 2012, Solar Physics 275, 17)
    flux = pd.to_numeric(sxr_flux, errors="coerce").ffill().bfill().clip(lower=0).fillna(0)
    em_proxy = np.sqrt(flux)
    em_series = pd.Series(em_proxy, index=flux.index)
    em_growth = em_series.diff().rolling(window=2, min_periods=1).mean().clip(lower=0)
    norm = em_growth[em_growth > 0].quantile(0.95)
    if not np.isfinite(norm) or norm <= 0:
        norm = max(float(em_growth.max()), 1e-12)
    g_component = (em_growth / norm).clip(0, 1)

    p_component = (g_component > 0.1).rolling(window=10, min_periods=1).mean().clip(0, 1)

    duration_minutes = _minutes_since_c1(flux) * cadence_minutes
    d_component = (duration_minutes / 60.0).clip(0, 1)

    derivative = flux.diff().fillna(0).abs()
    quiet_window = max(1, int(len(flux) * 0.2))
    quiet_mask = flux.iloc[:quiet_window] < C1_FLUX
    quiet_derivative = derivative.iloc[:quiet_window][quiet_mask]
    sigma_max = quiet_derivative.std()
    if not np.isfinite(sigma_max) or sigma_max <= 0:
        sigma_max = derivative.quantile(0.95)
    if not np.isfinite(sigma_max) or sigma_max <= 0:
        sigma_max = 1e-12
    s_component = (1 - (derivative / sigma_max).clip(0, 1)).clip(0, 1)

    fai = 0.3 * g_component + 0.3 * p_component + 0.2 * d_component + 0.2 * s_component
    return fai.clip(0, 1).fillna(0)


def minutes_since_c1_threshold(sxr_flux: pd.Series, cadence_minutes: int = 1) -> pd.Series:
    return _minutes_since_c1(pd.to_numeric(sxr_flux, errors="coerce").fillna(0)) * cadence_minutes
