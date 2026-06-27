import pandas as pd

from src.nri import compute_nri, quiet_sun_sigma


def test_nri_positive_during_flare():
    sxr = pd.Series([5e-7] * 20 + [1e-6, 2e-6, 4e-6, 6e-6, 5e-6])
    hxr = pd.Series([0.01] * 20 + [0.1, 0.4, 1.0, 0.8, 0.2])
    nri, _ = compute_nri(hxr, sxr, k=0.4)
    assert nri.iloc[-3:].max() > 0


def test_nri_near_zero_quiet_sun():
    sxr = pd.Series([5e-7] * 30)
    hxr = pd.Series([0.02] * 30)
    nri, _ = compute_nri(hxr, sxr)
    sigma = quiet_sun_sigma(nri, sxr)
    assert abs(nri.mean()) <= sigma
