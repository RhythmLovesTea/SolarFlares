from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.download_fermi import synthetic_hxr_from_goes
from data.download_goes import build_sample_event, fetch_realtime_json
from src.fai import compute_fai
from src.ladder import replay_ladder
from src.metrics import compute_metrics
from src.nowcasting import build_event_labels, detect_flare_events, save_events_sqlite
from src.nri import compute_nri, nri_sigma_series, quiet_sun_sigma


def run_pipeline(input_csv: str | None = None) -> pd.DataFrame:
    if input_csv and Path(input_csv).exists():
        data = pd.read_csv(input_csv, parse_dates=["time_tag"]).set_index("time_tag")
    else:
        sample_path = Path("data/sample_event.csv")
        data = pd.read_csv(sample_path, parse_dates=["time_tag"]).set_index("time_tag") if sample_path.exists() else build_sample_event()
    data["hxr_proxy"] = synthetic_hxr_from_goes(data["sxr_a"], data["sxr_b"])
    data["fai"] = compute_fai(data["sxr_b"])
    data["nri"], k = compute_nri(data["hxr_proxy"], data["sxr_b"])
    sigma = quiet_sun_sigma(data["nri"], data["sxr_b"])
    data["nri_sigma"] = data["nri"] / sigma
    readings = replay_ladder(data["fai"], data["nri_sigma"])
    data["ladder_state"] = [reading.state.name for reading in readings]
    events = detect_flare_events(data["sxr_b"], data["nri"], sigma, data["fai"])
    save_events_sqlite(events)
    labels = build_event_labels(data.index, events)
    predictions = data["ladder_state"].isin(["RED", "CRITICAL"]).astype(int)
    metrics = compute_metrics(predictions, labels, [event.get("lead_time_min", 0) for event in events.to_dict("records")])
    print(f"Computed FAI/NRI with k={k:.3f}, sigma={sigma:.4f}")
    print(f"Detected {len(events)} flare event(s)")
    print(metrics)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="AgniDrishti CLI pipeline")
    parser.add_argument("--input", default=None)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()
    if args.realtime:
        frame = fetch_realtime_json()
        path = Path("data/cache/goes_realtime.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path)
        args.input = str(path)
    run_pipeline(args.input)


if __name__ == "__main__":
    main()
