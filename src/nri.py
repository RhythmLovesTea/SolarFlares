from __future__ import annotations

import numpy as np
import pandas as pd

from .fai import C1_FLUX
from .preprocessing import savitzky_golay_derivative


def normalize_unit(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").ffill().bfill().fillna(0)
    minimum = values.min()
    maximum = values.max()
    if not np.isfinite(maximum - minimum) or maximum <= minimum:
        return pd.Series(0.0, index=values.index)
    return ((values - minimum) / (maximum - minimum)).clip(0, 1)


def estimate_coupling(hxr_norm: pd.Series, sxr_derivative_norm: pd.Series) -> float:
    rise_mask = sxr_derivative_norm > sxr_derivative_norm.quantile(0.75)
    x = sxr_derivative_norm[rise_mask].to_numpy(dtype=float)
    y = hxr_norm[rise_mask].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if valid.sum() < 3:
        return 1.0
    return float(np.clip(np.dot(x[valid], y[valid]) / np.dot(x[valid], x[valid]), 0.1, 5.0))


def compute_nri(
    hxr_flux: pd.Series,
    sxr_flux: pd.Series,
    k: float | None = None,
    input_kind: str | None = None,
    quiet_mask: pd.Series | None = None,
) -> tuple[pd.Series, float]:
    """
    Neupert Residual Index -- non-thermal ignition signal from HEL1OS.
    Uses Fermi GBM 25-50 keV as HEL1OS proxy.

    Physics: NRI(t) = F_HXR(t) - k * dF_SXR/dt(t)

    Citations:
    - Neupert effect: Neupert (1968), ApJ 153, L59
    - Statistical validation: Veronig et al. (2005), A&A 431, 1047
    - NRI operationalisation: AgniDrishti proposal, Team HelioDynamics, BAH 2026
    """
    # Fermi GBM HXR proxy: Meegan et al. (2009), ApJ 702, 791
    del input_kind
    sxr = pd.to_numeric(sxr_flux, errors="coerce").ffill().bfill().fillna(0)
    hxr = pd.to_numeric(hxr_flux, errors="coerce").reindex(sxr.index).interpolate().ffill().bfill().fillna(0)
    hxr_norm = normalize_unit(hxr)
    derivative = savitzky_golay_derivative(sxr).clip(lower=0)
    derivative_norm = normalize_unit(derivative)
    coupling = estimate_coupling(hxr_norm, derivative_norm) if k is None else float(k)
    nri = hxr_norm - coupling * derivative_norm
    quiet = pd.Series(quiet_mask, index=sxr.index).fillna(False).astype(bool) if quiet_mask is not None else sxr < C1_FLUX
    if quiet.any():
        nri = nri - nri[quiet].median()
    return nri.fillna(0), coupling


def quiet_sun_sigma(
    nri: pd.Series,
    sxr_flux: pd.Series,
    input_kind: str | None = None,
    quiet_mask: pd.Series | None = None,
) -> float:
    del input_kind
    sxr = pd.to_numeric(sxr_flux, errors="coerce").reindex(nri.index).ffill().bfill().fillna(0)
    if quiet_mask is not None:
        quiet_selector = pd.Series(quiet_mask, index=sxr.index).fillna(False).astype(bool)
        quiet = nri[quiet_selector]
    else:
        quiet = nri[sxr < C1_FLUX]
    sigma = quiet.std()
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = nri.iloc[: max(3, int(len(nri) * 0.2))].std()
    positive_tail = nri.clip(lower=0)
    proxy_floor = positive_tail.quantile(0.999) / 18 if positive_tail.max() > 0 else 0
    if np.isfinite(proxy_floor) and proxy_floor > 0:
        sigma = max(float(sigma), float(proxy_floor))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = max(float(nri.std()), 1e-6)
    return float(sigma)


def nri_sigma_series(nri: pd.Series, sxr_flux: pd.Series, sigma_quiet: float | None = None) -> pd.Series:
    sigma = quiet_sun_sigma(nri, sxr_flux) if sigma_quiet is None else float(sigma_quiet)
    sigma = max(sigma, 1e-12)
    return (nri / sigma).replace([np.inf, -np.inf], 0).fillna(0)
