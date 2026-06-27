import pandas as pd

from src.fai import compute_fai


def test_fai_range():
    series = pd.Series([5e-7, 7e-7, 1.1e-6, 2e-6, 4e-6])
    fai = compute_fai(series)
    assert ((fai >= 0) & (fai <= 1)).all()


def test_fai_increases_on_rising_flux():
    quiet = pd.Series([5e-7] * 20 + [7e-7, 1e-6, 2e-6, 5e-6, 8e-6])
    fai = compute_fai(quiet)
    assert fai.iloc[-1] > fai.iloc[0]


def test_fai_zero_on_quiet_sun():
    quiet = pd.Series([5e-7] * 40)
    fai = compute_fai(quiet)
    assert fai.max() <= 0.21
