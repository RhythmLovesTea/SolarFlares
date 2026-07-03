
# Coronalytics / AgniDrishti

Physics-informed solar flare nowcasting and event characterization for Aditya-L1, with deterministic fallback paths for offline benchmarking on GOES and Fermi proxy data.

AgniDrishti is designed as a science-facing prototype rather than a pure dashboard demo. It combines soft X-ray thermal preconditioning, hard X-ray ignition sensing, an interpretable instability ladder, event cataloguing, and instrument-specific detection logic into one reproducible pipeline.

## Why this project matters

Solar flare forecasting is difficult because the most useful signals are not always obvious in a single channel. This project addresses that problem by fusing complementary physical cues:

- thermal buildup in soft X-rays
- impulsive energy release in hard X-rays
- Neupert-effect-aware timing between the two
- state transitions that remain interpretable to domain reviewers

The result is a system that is easier to justify than a black-box classifier and more operational than a purely descriptive plot.

## Core ideas

### 1. Flare Anticipation Index `FAI`

`FAI` captures thermal preconditioning from soft X-ray evolution. It is built from four normalized components:

- growth of the emission proxy
- persistence above a growth threshold
- duration above a physical threshold
- suppression of high-derivative noise during quiet intervals

The implementation is in [`src/fai.py`](src/fai.py).

### 2. Neupert Residual Index `NRI`

`NRI` measures hard X-ray activity after accounting for the soft X-ray derivative term expected from the Neupert effect:

```text
NRI(t) = F_HXR(t) - k * dF_SXR/dt(t)
```

The coupling factor `k` is estimated from rise-phase samples unless explicitly provided. The index is then normalized into quiet-Sun sigma units for downstream logic.

The implementation is in [`src/nri.py`](src/nri.py).

### 3. Five-stage instability ladder

The system maps `FAI` and `NRI` into a monotonic state machine:

1. `QUIET`
2. `PREHEATING`
3. `THERMAL_INSTABILITY`
4. `IGNITION`
5. `CRITICAL`

This makes the output human-auditable and prevents unrealistic jumps between distant states.

The implementation is in [`src/ladder.py`](src/ladder.py).

### 4. Event detection and catalogue export

The detector uses dual-trigger logic:

- soft trigger: soft X-ray above threshold and rising
- hard trigger: `NRI` above a sigma threshold

It then groups nearby triggers, estimates start/peak/end times, assigns a GOES-like flare class, and can persist the result to SQLite.

The implementation is in [`src/nowcasting.py`](src/nowcasting.py).

## Scientific framing

The design is guided by physically interpretable relationships rather than raw pattern matching:

- the thermal channel reflects plasma heating and emission-measure growth
- the hard X-ray channel reflects impulsive non-thermal electron acceleration
- the Neupert effect links the two in a way that is useful for flare timing
- the ladder formalizes those relationships into operational states

That makes the system suitable for evaluator-facing demonstrations where explainability matters as much as accuracy.

## Data modes

AgniDrishti supports multiple execution paths depending on what data are available.

### Aditya-L1 live ingestion

If raw FITS products are available locally, the app can ingest SoLEXS and HEL1OS data directly through the Aditya-L1 loader in [`src/aditya_l1.py`](src/aditya_l1.py).

### Cached Aditya-L1 snapshot

If FITS files are unavailable, the app can use a pre-exported parquet snapshot stored under `data/cache/`.

### Committed Aditya-L1 summary snapshot

For fully offline evaluation, the repository includes a committed JSON snapshot at [`data/aditya_l1_snapshot.json`](data/aditya_l1_snapshot.json). This allows evaluators to inspect a real-data result path even without downloading the full archive.

### GOES/Fermi proxy mode

If Aditya-L1 data are not available, the pipeline can replay a proxy dataset built from:

- GOES XRS soft X-ray data
- Fermi GBM hard X-ray counts

This path is useful for offline reproducibility, debugging, and benchmarking.

## Independent instrument characterization

In addition to the fused physics pipeline, the repository includes a second analysis track in [`src/independent_characterization.py`](src/independent_characterization.py):

- `characterize_solexs()` for thermal flare detection on SoLEXS
- `characterize_hel1os()` for hard X-ray burst detection on HEL1OS
- `fuse_independent_detections()` for Neupert-effect-aware temporal matching

This is useful when you want to inspect each instrument separately before fusing them into a joint catalogue.

## Benchmark metrics

The project computes standard forecasting metrics in [`src/metrics.py`](src/metrics.py), including:

- `TPR` / `POD`
- `FAR`
- `TSS`
- `HSS`
- `CSI`
- `AUC-ROC`
- median lead time and interquartile range
- class-wise performance by GOES-like bucket

Important limitation:

- the committed real Aditya-L1 snapshot is a short window, not a full archive
- reported benchmark-style scores are best interpreted as prototype-level evaluation unless you rerun the pipeline on a broader dataset

- <img width="1895" height="703" alt="image" src="https://github.com/user-attachments/assets/35caaf28-c447-4004-b8b5-4bb554a52c13" />
  <img width="1909" height="844" alt="image" src="https://github.com/user-attachments/assets/42b4d6e2-67bf-4c0d-b863-e5e97e726f59" />
  <img width="1920" height="747" alt="image" src="https://github.com/user-attachments/assets/4d8ecf22-38b3-419e-bf3a-8b17931e4576" />


## Repository architecture

```text
AgniDrishti/
├── app.py                     # Streamlit dashboard
├── main.py                    # CLI pipeline entry point
├── data/
│   ├── download_goes.py       # GOES downloader + sample event generator
│   ├── download_fermi.py      # Fermi GBM downloader + proxy builder
│   ├── fetch_aditya_l1.py     # Aditya-L1 fetch helper
│   └── aditya_l1_snapshot.json
├── src/
│   ├── aditya_l1.py           # Aditya-L1 ingestion and snapshot support
│   ├── fai.py                 # Flare Anticipation Index
│   ├── nri.py                 # Neupert Residual Index
│   ├── ladder.py              # Instability ladder state machine
│   ├── nowcasting.py          # Flare detection and catalogue export
│   ├── metrics.py             # Evaluation metrics
│   ├── preprocessing.py       # Signal processing helpers
│   ├── visualisation.py       # Plotly visualisations
│   ├── forecasting.py         # Lightweight calibration helpers
│   └── independent_characterization.py
├── scripts/
│   └── export_fits_to_parquet.py
└── tests/
```

## How it works end to end

1. Load soft X-ray and hard X-ray time series from Aditya-L1 or a proxy source.
2. Compute `FAI` from soft X-ray thermal evolution.
3. Compute `NRI` from hard X-ray activity minus the expected thermal derivative term.
4. Normalize `NRI` against the quiet-Sun baseline.
5. Replay the instability ladder across the time axis.
6. Detect flare events using combined thermal and hard-X-ray triggers.
7. Export the catalogue and calculate metrics for evaluation.

## Running the project

###You can directly Access the Deployed Dashboard by clicking on This Link !!!
https://coronalytics.streamlit.app/

Otherwise Manually ---

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch the Streamlit dashboard

```bash
streamlit run app.py
```

### Run the CLI pipeline

```bash
python3 main.py
```

### Run on a local CSV input

```bash
python3 main.py --input data/sample_event.csv
```

### Run with realtime GOES input

```bash
python3 main.py --realtime
```

## Testing

The repository includes focused unit tests for the physical indices and ladder behavior:

- [`tests/test_fai.py`](tests/test_fai.py)
- [`tests/test_nri.py`](tests/test_nri.py)
- [`tests/test_ladder.py`](tests/test_ladder.py)

Run them with:

```bash
pytest
```

## Key outputs

The pipeline can produce:

- per-timestamp `FAI`, `NRI`, and ladder state
- flare event tables with start, peak, and end times
- GOES-like flare class estimates
- lead-time statistics
- SQLite catalogue output
- dashboard visualisations for model review

## Data and reproducibility notes

- The repo is structured to run even when the full Aditya-L1 archive is unavailable.
- The offline snapshot and proxy modes are intentionally included to make the project reviewable on limited infrastructure.
- The design favors traceable rules and measurable thresholds so that results can be defended during evaluation.

## Dependencies

The project uses:

- `pandas`, `numpy`, `scipy` for signal processing
- `scikit-learn` for metrics and lightweight calibration
- `streamlit` and `plotly` for the UI
- `astropy`, `sunpy`, `netCDF4`, and `requests` for solar-data workflows
- `sqlite3` for event catalogue persistence

See [`requirements.txt`](requirements.txt) for the full list.

## Suggested reviewer narrative

If you are presenting this project to evaluators, the strongest framing is:

- multi-instrument solar flare intelligence
- physics-informed and interpretable rather than opaque
- capable of live, cached, and offline operation
- backed by event detection, cataloguing, and quantitative metrics
- validated with unit tests around the core scientific logic

## License

Add the appropriate license for your repository if it is not already present.
