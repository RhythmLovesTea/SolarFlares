#!/usr/bin/env python
"""
scripts/export_fits_to_parquet.py
──────────────────────────────────
Export the local Aditya-L1 FITS data to a compressed Parquet snapshot
and a GTI JSON sidecar so they can be committed to GitHub and loaded on
Streamlit Community Cloud without the raw 290 MB FITS archive.

Usage (from repo root):
    python scripts/export_fits_to_parquet.py
"""
from __future__ import annotations

import json
import os
import sys

# Ensure the repo root is on sys.path so src/ imports work.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pandas as pd

from src.aditya_l1 import load_aditya_l1_dataset

DATA_ROOT   = os.path.join(REPO_ROOT, "data", "aditya_l1")
CACHE_DIR   = os.path.join(REPO_ROOT, "data", "cache")
PARQUET_OUT = os.path.join(CACHE_DIR, "aditya_l1_realdata.parquet")
GTI_OUT     = os.path.join(CACHE_DIR, "aditya_l1_gti.json")


def _gti_df_to_list(gti) -> list[list[str]]:
    """Convert a GTI DataFrame (start/stop cols) or list-of-tuples to ISO strings."""
    result = []
    if hasattr(gti, "iterrows"):
        # DataFrame with 'start' / 'stop' columns
        for _, row in gti.iterrows():
            s = pd.Timestamp(row["start"])
            e = pd.Timestamp(row["stop"])
            result.append([s.isoformat(), e.isoformat()])
    else:
        # list of (start, end) tuples
        for s, e in gti:
            result.append([pd.Timestamp(s).isoformat(), pd.Timestamp(e).isoformat()])
    return result


def main() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"Loading Aditya-L1 dataset from: {DATA_ROOT}")
    frame, products = load_aditya_l1_dataset(DATA_ROOT)

    # ── 1. Save Parquet ────────────────────────────────────────────────────────
    # Ensure the index is UTC-aware before writing (pyarrow requires it).
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")

    frame.to_parquet(PARQUET_OUT, compression="snappy", engine="pyarrow")
    parquet_size_mb = os.path.getsize(PARQUET_OUT) / 1_048_576
    print(f"Saved parquet : {parquet_size_mb:.1f} MB, {len(frame):,} rows  →  {PARQUET_OUT}")

    # ── 2. Build GTI JSON ──────────────────────────────────────────────────────
    sdd2_gti = products.get("sdd2_gti")
    czt_gti  = products.get("czt_gti")

    gti_data: dict = {
        "sdd2_gti": _gti_df_to_list(sdd2_gti) if sdd2_gti is not None else [],
        "czt_gti":  _gti_df_to_list(czt_gti)  if czt_gti  is not None else [],
        "notices":  products.get("notices", []),
    }

    with open(GTI_OUT, "w") as fh:
        json.dump(gti_data, fh, indent=2)

    gti_size_kb = os.path.getsize(GTI_OUT) / 1024
    print(f"Saved GTI JSON: {gti_size_kb:.1f} KB  →  {GTI_OUT}")

    # ── 3. Summary ─────────────────────────────────────────────────────────────
    print()
    print("GTI intervals written:")
    print(f"  sdd2_gti: {len(gti_data['sdd2_gti'])} interval(s)")
    print(f"  czt_gti : {len(gti_data['czt_gti'])} interval(s)")
    if gti_data["notices"]:
        print("Notices carried over:")
        for n in gti_data["notices"]:
            print(f"  • {n}")

    print()
    if parquet_size_mb < 50:
        print(f"✅ Parquet is {parquet_size_mb:.1f} MB — safe to commit to GitHub (<50 MB limit).")
    else:
        print(f"⚠️  Parquet is {parquet_size_mb:.1f} MB — may exceed GitHub's 50 MB file limit!")
        print("   Consider using Git LFS or further compressing before committing.")


if __name__ == "__main__":
    main()
