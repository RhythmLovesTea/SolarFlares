from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .fai import compute_fai, minutes_since_c1_threshold
from .ladder import replay_ladder
from .nri import compute_nri, nri_sigma_series
from .nowcasting import build_event_labels


class FlareCalibrator:
    """
    Lightweight ML calibrator trained on physics features.
    Maps (FAI, NRI, ladder_state, dFAI/dt, dNRI/dt) to P(flare >= M within 30 min).

    Citation:
    - Bobra & Couvidat (2015), ApJ 798, 135 -- SVM for flare prediction
    - Bloomfield et al. (2012), ApJ 747, 41 -- operational forecasting benchmarks
    - Hassani et al. (2025), ApJS 279, 27 -- LSTM benchmark (TSS=0.74, AUC=0.87)
    """

    def __init__(self, model_path: str | Path = "models/calibrator.pkl") -> None:
        self.model_path = Path(model_path)
        base = RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
        )
        self.model = CalibratedClassifierCV(base, method="isotonic", cv=3)

    def build_training_dataset(self, sxr_series, hxr_series, flare_catalogue_df):
        sxr = pd.to_numeric(sxr_series, errors="coerce").ffill().bfill().fillna(0)
        hxr = pd.to_numeric(hxr_series, errors="coerce").reindex(sxr.index).interpolate().ffill().bfill().fillna(0)
        fai = compute_fai(sxr)
        nri, _ = compute_nri(hxr, sxr)
        nri_sigma = nri_sigma_series(nri, sxr)
        readings = replay_ladder(fai, nri_sigma)
        ladder_state_int = pd.Series([reading.state.value for reading in readings], index=sxr.index)
        features = pd.DataFrame(
            {
                "FAI": fai,
                "NRI_sigma": nri_sigma,
                "ladder_state_int": ladder_state_int,
                "fai_derivative": fai.diff().fillna(0),
                "nri_derivative": nri_sigma.diff().fillna(0),
                "fai_10min_max": fai.rolling(10, min_periods=1).max(),
                "nri_10min_max": nri_sigma.rolling(10, min_periods=1).max(),
                "minutes_since_c1_threshold": minutes_since_c1_threshold(sxr),
            }
        ).replace([np.inf, -np.inf], 0).fillna(0)
        labels = build_event_labels(features.index, flare_catalogue_df)
        return features, labels

    def train(self, X, y):
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2 or len(y) < 10:
            fallback = LogisticRegression(class_weight="balanced", max_iter=1000)
            fallback.fit(X, y if len(np.unique(y)) == 2 else np.r_[y[:-1], 1])
            self.model = fallback
            report = {"AUC_ROC": 0.0, "TSS": 0.0, "FAR": 0.0}
        else:
            cv = StratifiedKFold(n_splits=min(3, np.bincount(y).min()), shuffle=True, random_state=42)
            probas = cross_val_predict(self.model, X, y, cv=cv, method="predict_proba")[:, 1]
            preds = (probas >= 0.5).astype(int)
            tp = ((preds == 1) & (y == 1)).sum()
            fp = ((preds == 1) & (y == 0)).sum()
            tn = ((preds == 0) & (y == 0)).sum()
            fn = ((preds == 0) & (y == 1)).sum()
            pod = tp / (tp + fn) if tp + fn else 0.0
            pofd = fp / (fp + tn) if fp + tn else 0.0
            report = {
                "AUC_ROC": float(roc_auc_score(y, probas)),
                "TSS": float(pod - pofd),
                "FAR": float(fp / (tp + fp)) if tp + fp else 0.0,
            }
            self.model.fit(X, y)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        return report

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        self.model = joblib.load(self.model_path)
        return True

    def predict_proba(self, fai, nri_sigma, ladder_state, fai_derivative=0, nri_derivative=0, **kwargs):
        row = pd.DataFrame(
            [
                {
                    "FAI": fai,
                    "NRI_sigma": nri_sigma,
                    "ladder_state_int": getattr(ladder_state, "value", ladder_state),
                    "fai_derivative": fai_derivative,
                    "nri_derivative": nri_derivative,
                    "fai_10min_max": kwargs.get("fai_10min_max", fai),
                    "nri_10min_max": kwargs.get("nri_10min_max", nri_sigma),
                    "minutes_since_c1_threshold": kwargs.get("minutes_since_c1_threshold", 0),
                }
            ]
        )
        if hasattr(self.model, "predict_proba"):
            mplus = float(self.model.predict_proba(row)[0][-1])
        else:
            mplus = float(np.clip(0.5 * fai + 0.1 * nri_sigma, 0, 1))
        xclass = float(np.clip(mplus * max(0, fai - 0.7) * 1.4, 0, 1))
        cclass = float(np.clip(1 - mplus, 0, 1))
        return {"C-class": cclass, "M-class": mplus, "X-class": xclass}
