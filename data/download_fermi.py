from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests


FERMI_BASE = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily"


def ctime_url(day: datetime, detector: str = "n5") -> str:
    return f"{FERMI_BASE}/{day:%Y/%m/%d}/current/glg_ctime_{detector}_{day:%y%m%d}_v00.pha"


def download_fermi_ctime(day: str, detector: str = "n5", cache_dir: Path = Path("data/cache")) -> Path:
    # Fermi GBM HXR proxy: Meegan et al. (2009), ApJ 702, 791
    dt = pd.to_datetime(day).to_pydatetime()
    url = ctime_url(dt, detector)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / url.rsplit("/", 1)[-1]
    if target.exists() and target.stat().st_size > 0:
        return target
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def parse_fermi_ctime(path: Path) -> pd.DataFrame:
    from astropy.io import fits

    with fits.open(path) as hdul:
        data = hdul["RATE"].data
        trigtime = hdul["PRIMARY"].header.get("TRIGTIME", 0.0)
        times = np.asarray(data["TIME"], dtype=float) + float(trigtime)
        rates = np.asarray(data["RATE"], dtype=float)
        channel_index = min(2, rates.shape[1] - 1)
        counts = rates[:, channel_index]
    epoch = pd.Timestamp("2001-01-01T00:00:00Z")
    index = epoch + pd.to_timedelta(times, unit="s")
    frame = pd.DataFrame({"hxr_proxy": counts}, index=index).resample("1min").mean().interpolate()
    frame.index.name = "time_tag"
    return frame


def synthetic_hxr_from_goes(sxr_a: pd.Series, sxr_b: pd.Series) -> pd.Series:
    derivative = sxr_a.fillna(sxr_b).diff().clip(lower=0).fillna(0)
    scaled = derivative / max(float(derivative.quantile(0.99)), 1e-12)
    microbursts = sxr_b.diff().clip(lower=0).rolling(3, min_periods=1).max()
    microbursts = microbursts / max(float(microbursts.quantile(0.99)), 1e-12)
    return (0.65 * scaled + 0.35 * microbursts).clip(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Fermi GBM CTIME data.")
    parser.add_argument("--date", default="2022-03-28")
    parser.add_argument("--detector", default="n5")
    parser.add_argument("--output", default="data/cache/fermi_ctime.csv")
    args = parser.parse_args()
    path = download_fermi_ctime(args.date, args.detector)
    frame = parse_fermi_ctime(path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output)
    print(f"Wrote {len(frame)} Fermi rows to {output}")


if __name__ == "__main__":
    main()
