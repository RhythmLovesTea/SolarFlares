"""fetch_aditya_l1.py

Downloads the Aditya-L1 Level-1 FITS archive (SoLEXS + HEL1OS) from
Hugging Face Hub into data/aditya_l1/.

Usage
-----
    python data/fetch_aditya_l1.py

Requirements
------------
    pip install huggingface_hub

Notes
-----
- Total download ~125 MB (lightcurves + GTI files only; spectra excluded)
- Run once; subsequent runs are instant (files already present)
- If you skip this step the app automatically falls back to the
  pre-computed snapshot (data/aditya_l1_snapshot.json) which still
  shows real SoLEXS/HEL1OS detection results in the instrument tabs.

Data source
-----------
Aditya-L1 Level-1 data are publicly released by ISRO via PRADAN
(Physical Research Data Archive and Node):
    https://pradan.issdc.gov.in/

This Hugging Face mirror contains only the lightcurve and GTI products
used by Coronalytics (SoLEXS SDD2, HEL1OS CZT/CdTe); raw spectra and
event lists are excluded to keep the download manageable.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ID = "RhythmLovesTea/SolarFlares-Data"   # <-- set after HF upload
_LOCAL_DIR = Path(__file__).parent / "aditya_l1"
_SENTINEL = _LOCAL_DIR / "AL1_SLX_L1_20260625_v1.0" / "SDD2" / "AL1_SOLEXS_20260625_SDD2_L1.lc.gz"


def _check_already_present() -> bool:
    return _SENTINEL.exists()


def _download() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub not installed.\n"
            "Run:  pip install huggingface_hub\n"
            "Then re-run this script."
        )
        sys.exit(1)

    print(f"Downloading Aditya-L1 FITS archive from {_REPO_ID} ...")
    print("(~125 MB, run once only)\n")

    snapshot_download(
        repo_id=_REPO_ID,
        repo_type="dataset",
        local_dir=str(_LOCAL_DIR),
        ignore_patterns=[
            # Exclude large spectra and event-list files (~170 MB)
            "*.pi.gz",
            "*spectra*.fits",
            "events/evt.fits",
        ],
    )
    print(f"\n✅  Data downloaded to {_LOCAL_DIR.resolve()}")


def main() -> None:
    if _check_already_present():
        print(f"✅  Aditya-L1 data already present at {_LOCAL_DIR.resolve()}")
        print("    Nothing to download.")
        return
    _download()


if __name__ == "__main__":
    main()
