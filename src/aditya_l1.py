from __future__ import annotations

import glob
import gzip
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
from astropy.io import fits
from astropy.time import Time

__all__ = ["diagnose_data_root", "load_aditya_l1_dataset"]


def _glob_paths(data_root: str, pattern: str) -> list[str]:
    return sorted(glob.glob(os.path.join(data_root, "**", pattern), recursive=True))


def _first_path(paths: list[str]) -> str | None:
    return paths[0] if paths else None


def _open_gz_fits(path: str):
    with gzip.open(path, "rb") as handle:
        payload = handle.read()
    return fits.open(BytesIO(payload))


def _read_solexs_lightcurve(path: str | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["counts"], index=pd.DatetimeIndex([], tz="UTC"))
    with _open_gz_fits(path) as hdul:
        rate = hdul["RATE"].data
        index = pd.to_datetime(rate["TIME"], unit="s", utc=True)
        frame = pd.DataFrame({"counts": rate["COUNTS"].astype("float64")}, index=index)
    frame.index.name = "time"
    return frame.sort_index()


def _read_solexs_gti(path: str | None) -> pd.DataFrame:
    columns = ["start", "stop"]
    if path is None:
        return pd.DataFrame(columns=columns)
    with _open_gz_fits(path) as hdul:
        gti = hdul["GTI"].data
        if len(gti) == 0:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(
            {
                "start": pd.to_datetime(gti["START"], unit="s", utc=True),
                "stop": pd.to_datetime(gti["STOP"], unit="s", utc=True),
            }
        )


def _select_hdu_by_token(hdul, token: str):
    for hdu in hdul[1:]:
        if token in hdu.name:
            return hdu
    return None


def _read_hel1os_fullband_series(path: str, fullband_token: str, prefix: str) -> pd.DataFrame:
    with fits.open(path) as hdul:
        hdu = _select_hdu_by_token(hdul, fullband_token)
        if hdu is None:
            raise ValueError(f"No HEL1OS HDU containing {fullband_token!r} found in {path}")
        data = hdu.data
        isot_values = [str(value).strip() for value in data["ISOT"]]
        index = pd.to_datetime(isot_values, utc=True)
        detector = Path(path).stem.split("_")[-1]
        frame = pd.DataFrame(
            {
                f"{prefix}_{detector}_ctr": data["CTR"].astype("float64"),
                f"{prefix}_{detector}_stat_err": data["STAT_ERR"].astype("float64"),
            },
            index=index,
        )
    frame.index.name = "time"
    return frame.sort_index()


def _merge_hel1os_lightcurves(paths: list[str], fullband_token: str, prefix: str) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    merged = pd.concat(
        [_read_hel1os_fullband_series(path, fullband_token, prefix) for path in paths],
        axis=1,
        sort=True,
    ).sort_index()
    ctr_columns = [column for column in merged.columns if column.endswith("_ctr")]
    if ctr_columns:
        merged[f"{prefix}_fullband_ctr"] = merged[ctr_columns].mean(axis=1)
    return merged


def _read_hel1os_gti(paths: list[str], source_prefix: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in paths:
        with fits.open(path) as hdul:
            gti = hdul[1].data
            if len(gti) == 0:
                continue
            mjd_times = Time(gti["tstart"], format="mjd").to_datetime()
            mjd_stops = Time(gti["tstop"], format="mjd").to_datetime()
            detector = Path(path).stem.replace("gti", "", 1)
            rows.append(
                pd.DataFrame(
                    {
                        "start": pd.to_datetime(mjd_times, utc=True),
                        "stop": pd.to_datetime(mjd_stops, utc=True),
                        "detector": detector,
                        "source": source_prefix,
                    }
                )
            )
    if not rows:
        return pd.DataFrame(columns=["start", "stop", "detector", "source"])
    return pd.concat(rows, ignore_index=True)


def diagnose_data_root(data_root: str) -> list[str]:
    root = str(Path(data_root))
    sdd1_lc = _glob_paths(root, "*SDD1*L1.lc.gz")
    sdd1_gti = _glob_paths(root, "*SDD1*L1.gti.gz")
    sdd2_lc = _glob_paths(root, "*SDD2*L1.lc.gz")
    sdd2_gti = _glob_paths(root, "*SDD2*L1.gti.gz")
    cdte_lc = _glob_paths(root, "lightcurve_cdte*.fits")
    czt_lc = _glob_paths(root, "lightcurve_czt*.fits")
    cdte_gti = _glob_paths(root, "gticdte*.fits")
    czt_gti = _glob_paths(root, "gticzt*.fits")
    hel1os_events = _glob_paths(root, "evt.fits")

    lines = [
        f"✅ Data root found: {Path(root).resolve()}" if Path(root).exists() else f"❌ Data root missing: {Path(root).resolve()}",
        f"✅ SoLEXS SDD2 lightcurve found: {sdd2_lc[0]}" if sdd2_lc else "❌ SoLEXS SDD2 lightcurve missing",
        f"✅ SoLEXS SDD2 GTI found: {sdd2_gti[0]}" if sdd2_gti else "❌ SoLEXS SDD2 GTI missing",
        (
            "⚠️ SoLEXS SDD1 lightcurve absent; GTI-only archive will be treated as unavailable"
            if not sdd1_lc and sdd1_gti
            else f"✅ SoLEXS SDD1 lightcurve found: {sdd1_lc[0]}"
            if sdd1_lc
            else "⚠️ SoLEXS SDD1 lightcurve missing"
        ),
        f"✅ SoLEXS SDD1 GTI found: {sdd1_gti[0]}" if sdd1_gti else "⚠️ SoLEXS SDD1 GTI missing",
        f"✅ HEL1OS CZT lightcurve(s) found: {len(czt_lc)} file(s)" if czt_lc else "❌ HEL1OS CZT lightcurve(s) missing",
        f"✅ HEL1OS CZT GTI file(s) found: {len(czt_gti)} file(s)" if czt_gti else "❌ HEL1OS CZT GTI file(s) missing",
        f"✅ HEL1OS CdTe lightcurve(s) found: {len(cdte_lc)} file(s)" if cdte_lc else "⚠️ HEL1OS CdTe lightcurve(s) missing",
        f"✅ HEL1OS CdTe GTI file(s) found: {len(cdte_gti)} file(s)" if cdte_gti else "⚠️ HEL1OS CdTe GTI file(s) missing",
        f"✅ HEL1OS event file found: {hel1os_events[0]}" if hel1os_events else "⚠️ HEL1OS event file missing",
    ]
    return lines


def load_aditya_l1_dataset(real_data_root: str) -> tuple[pd.DataFrame, dict]:
    root = str(Path(real_data_root))
    sdd1_lc_paths = _glob_paths(root, "*SDD1*L1.lc.gz")
    sdd1_gti_paths = _glob_paths(root, "*SDD1*L1.gti.gz")
    sdd2_lc_paths = _glob_paths(root, "*SDD2*L1.lc.gz")
    sdd2_gti_paths = _glob_paths(root, "*SDD2*L1.gti.gz")
    cdte_lc_paths = _glob_paths(root, "lightcurve_cdte*.fits")
    czt_lc_paths = _glob_paths(root, "lightcurve_czt*.fits")
    cdte_gti_paths = _glob_paths(root, "gticdte*.fits")
    czt_gti_paths = _glob_paths(root, "gticzt*.fits")
    hel1os_event_paths = _glob_paths(root, "evt.fits")

    if not sdd2_lc_paths:
        raise FileNotFoundError(f"SoLEXS SDD2 lightcurve not found under {Path(root).resolve()}")
    if not czt_lc_paths:
        raise FileNotFoundError(f"HEL1OS CZT lightcurve(s) not found under {Path(root).resolve()}")

    notices: list[str] = []

    sdd1 = _read_solexs_lightcurve(_first_path(sdd1_lc_paths))
    # Load and concatenate all available SDD2 lightcurve files chronologically.
    all_sdd2 = []
    all_sdd2_gti = []
    for lc_path in sdd2_lc_paths:
        date_tag = next(
            (part for part in os.path.basename(lc_path).split("_") if len(part) == 8 and part.isdigit()),
            "",
        )
        matching_gti = [gti_path for gti_path in sdd2_gti_paths if date_tag and date_tag in gti_path]
        all_sdd2.append(_read_solexs_lightcurve(lc_path))
        gti_frame = _read_solexs_gti(matching_gti[0] if matching_gti else None)
        if not gti_frame.empty:
            all_sdd2_gti.append(gti_frame)
    sdd2 = pd.concat(all_sdd2).sort_index()
    sdd2 = sdd2[~sdd2.index.duplicated(keep="first")]
    sdd1_gti = _read_solexs_gti(_first_path(sdd1_gti_paths))
    sdd2_gti = (
        pd.concat(all_sdd2_gti, ignore_index=True).sort_values("start").reset_index(drop=True)
        if all_sdd2_gti
        else _read_solexs_gti(None)
    )
    if not sdd1_lc_paths:
        notices.append("⚠️ SoLEXS SDD1 lightcurve absent; treating SDD1 as unavailable.")
    if sdd1_gti.empty:
        notices.append("⚠️ SoLEXS SDD1 GTI has 0 rows; SDD1 remains unavailable.")

    cdte = _merge_hel1os_lightcurves(cdte_lc_paths, "1.80KEV_TO_90", "cdte")
    czt = _merge_hel1os_lightcurves(czt_lc_paths, "18.00KEV_TO_160", "czt")
    cdte_gti = _read_hel1os_gti(cdte_gti_paths, "cdte")
    czt_gti = _read_hel1os_gti(czt_gti_paths, "czt")
    if len(czt_lc_paths) > 1:
        notices.append(f"✅ HEL1OS CZT full-band proxy built from {len(czt_lc_paths)} detector lightcurves.")
    if len(cdte_lc_paths) > 1:
        notices.append(f"✅ HEL1OS CdTe full-band products include {len(cdte_lc_paths)} detector lightcurves.")

    frame = pd.concat(
        [
            sdd2[["counts"]].rename(columns={"counts": "sxr_b"}),
            czt[[column for column in czt.columns if column == "czt_fullband_ctr"]].rename(
                columns={"czt_fullband_ctr": "hxr_proxy"}
            ),
        ],
        axis=1,
        sort=True,
    ).sort_index()
    frame = frame.resample("1s").mean()

    products = {
        "sdd1": sdd1,
        "sdd2": sdd2,
        "sdd1_gti": sdd1_gti,
        "sdd2_gti": sdd2_gti,
        "cdte": cdte,
        "czt": czt,
        "cdte_gti": cdte_gti,
        "czt_gti": czt_gti,
        "paths": {
            "sdd1_lightcurve": _first_path(sdd1_lc_paths),
            "sdd1_gti": _first_path(sdd1_gti_paths),
            "sdd2_lightcurve": _first_path(sdd2_lc_paths),
            "sdd2_gti": _first_path(sdd2_gti_paths),
            "hel1os_cdte_lightcurves": cdte_lc_paths,
            "hel1os_czt_lightcurves": czt_lc_paths,
            "hel1os_cdte_gti": cdte_gti_paths,
            "hel1os_czt_gti": czt_gti_paths,
            "hel1os_events": _first_path(hel1os_event_paths),
        },
        "notices": notices,
    }
    return frame, products
