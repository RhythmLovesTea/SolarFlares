from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests


NOAA_JSON_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
NOAA_NCEI_BASE = (
    "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/"
    "xrsf-l2-avg1m_science"
)


def fetch_realtime_json(cache_dir: Path = Path("data/cache")) -> pd.DataFrame:
    # Data: GOES XRS proxy for SoLEXS (Lemen et al. 2012, Solar Physics 275, 17)
    cache_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(NOAA_JSON_URL, timeout=30)
    response.raise_for_status()
    rows = response.json()
    (cache_dir / "xrays-7-day.json").write_text(json.dumps(rows), encoding="utf-8")
    frame = pd.DataFrame(rows)
    frame["time_tag"] = pd.to_datetime(frame["time_tag"], utc=True)
    pivot = frame.pivot_table(index="time_tag", columns="energy", values="flux", aggfunc="mean").sort_index()
    result = pd.DataFrame(index=pivot.index)
    result["sxr_a"] = pivot.get("0.05-0.4nm", pd.Series(index=pivot.index, dtype=float))
    result["sxr_b"] = pivot.get("0.1-0.8nm", pd.Series(index=pivot.index, dtype=float))
    result["source"] = "NOAA GOES primary 7-day JSON"
    return result.dropna(subset=["sxr_b"])


def _candidate_goes_urls(day: datetime) -> list[str]:
    stamp = day.strftime("%Y%m%d")
    return [
        f"{NOAA_NCEI_BASE}/{day:%Y/%m}/sci_xrsf-l2-avg1m_g16_d{stamp}_v2-2-1.nc",
        f"{NOAA_NCEI_BASE}/{day:%Y/%m}/sci_xrsf-l2-avg1m_g16_d{stamp}_v2-2-0.nc",
    ]


def download_historical_day(day: datetime, cache_dir: Path = Path("data/cache")) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for url in _candidate_goes_urls(day):
        target = cache_dir / url.rsplit("/", 1)[-1]
        if target.exists() and target.stat().st_size > 0:
            return target
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            target.write_bytes(response.content)
            return target
    raise FileNotFoundError(f"No GOES NetCDF found for {day:%Y-%m-%d}")


def parse_goes_netcdf(path: Path) -> pd.DataFrame:
    try:
        from scipy.io import netcdf_file

        with netcdf_file(path, "r", mmap=False) as nc:
            variables = nc.variables
            time_values = np.array(variables["time"][:], dtype=float)
            units = getattr(variables["time"], "units", b"seconds since 2000-01-01 12:00:00")
            if isinstance(units, bytes):
                units = units.decode("ascii", errors="ignore")
            epoch_text = units.split("since", 1)[1].strip().replace("Z", "")
            epoch = pd.to_datetime(epoch_text, utc=True)
            index = epoch + pd.to_timedelta(time_values, unit="s")
            a_name = "xrsa_flux" if "xrsa_flux" in variables else "xrsa_flux_observed"
            b_name = "xrsb_flux" if "xrsb_flux" in variables else "xrsb_flux_observed"
            result = pd.DataFrame(
                {
                    "sxr_a": np.array(variables[a_name][:], dtype=float),
                    "sxr_b": np.array(variables[b_name][:], dtype=float),
                    "source": path.name,
                },
                index=index,
            )
    except TypeError:
        from netCDF4 import Dataset, num2date

        with Dataset(path) as nc:
            variables = nc.variables
            time_var = variables["time"]
            times = num2date(time_var[:], time_var.units, only_use_cftime_datetimes=False)
            index = pd.to_datetime(times, utc=True)
            a_name = "xrsa_flux" if "xrsa_flux" in variables else "xrsa_flux_observed"
            b_name = "xrsb_flux" if "xrsb_flux" in variables else "xrsb_flux_observed"
            result = pd.DataFrame(
                {
                    "sxr_a": np.array(variables[a_name][:], dtype=float),
                    "sxr_b": np.array(variables[b_name][:], dtype=float),
                    "source": path.name,
                },
                index=index,
            )
    result.index.name = "time_tag"
    return result.replace(-9999, np.nan).dropna(subset=["sxr_b"])


def fetch_historical_range(start: str, end: str, cache_dir: Path = Path("data/cache")) -> pd.DataFrame:
    start_dt = pd.to_datetime(start).to_pydatetime()
    end_dt = pd.to_datetime(end).to_pydatetime()
    frames = []
    day = start_dt
    while day <= end_dt:
        try:
            frames.append(parse_goes_netcdf(download_historical_day(day, cache_dir)))
        except Exception as exc:
            print(f"GOES download skipped for {day:%Y-%m-%d}: {exc}")
        day += timedelta(days=1)
    if not frames:
        raise RuntimeError("No historical GOES files downloaded")
    return pd.concat(frames).sort_index()


def _warn_download_fallback(message: str) -> None:
    try:
        if "streamlit" not in sys.modules:
            print(message)
            return
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            st.warning(message)
        else:
            print(message)
    except Exception:
        print(message)


def load_goes_data(start_date, end_date=None, cache_dir="data/cache") -> pd.DataFrame:
    """
    Attempt to download GOES XRS data.
    Falls back to bundled sample if download fails.
    """
    try:
        if end_date is None:
            return fetch_realtime_json(Path(cache_dir))
        return fetch_historical_range(str(start_date), str(end_date), Path(cache_dir))
    except Exception as exc:
        _warn_download_fallback(
            f"⚠️ Live data download failed ({exc}). Loading bundled 2022-03-28 X1.5 event."
        )
        fallback_path = Path("data/sample_event.csv")
        if os.path.exists(fallback_path):
            return pd.read_csv(fallback_path, parse_dates=["time_tag"]).set_index("time_tag")
        raise RuntimeError("No data available. Run: python data/download_goes.py --sample") from exc


def build_sample_event(output: Path = Path("data/sample_event.csv")) -> pd.DataFrame:
    """Create a deterministic offline replay around the documented 2022-03-28 X1.5 flare."""
    output.parent.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp("2022-03-28T10:00:00Z")
    index = pd.date_range(start, periods=361, freq="min")
    minutes = np.arange(len(index), dtype=float)
    peak_minute = 178.0
    baseline = 7.0e-7 + 1.5e-7 * np.sin(minutes / 45)
    slow_rise = 2.2e-6 / (1 + np.exp(-(minutes - 90) / 18))
    impulsive = 1.25e-4 * np.exp(-0.5 * ((minutes - peak_minute) / 16) ** 2)
    decay = np.where(minutes > peak_minute, 2.5e-5 * np.exp(-(minutes - peak_minute) / 70), 0)
    sxr_b = np.maximum(baseline + slow_rise + impulsive + decay, 1e-9)
    sxr_a = np.maximum(0.08 * sxr_b + 2e-8 * np.exp(-0.5 * ((minutes - 166) / 8) ** 2), 1e-10)
    frame = pd.DataFrame(
        {
            "time_tag": index,
            "sxr_a": sxr_a,
            "sxr_b": sxr_b,
            "source": "offline replay calibrated to GOES class/timing for 2022-03-28 X1.5",
        }
    )
    frame.to_csv(output, index=False)
    return frame.set_index("time_tag")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GOES XRS data for AgniDrishti.")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--output", default="data/cache/goes_xrs.csv")
    args = parser.parse_args()
    if args.sample:
        frame = build_sample_event(Path("data/sample_event.csv"))
    elif args.realtime or not (args.start and args.end):
        frame = fetch_realtime_json()
    else:
        frame = fetch_historical_range(args.start, args.end)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output)
    print(f"Wrote {len(frame)} GOES rows to {output}")


if __name__ == "__main__":
    main()
