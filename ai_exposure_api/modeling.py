from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .config import (
    ABILITY_CONTRIBUTIONS_FILE,
    FEATURES_FILE,
    METRICS_FILE,
    MODEL_DIR,
    OCCUPATION_LOOKUP_FILE,
    PREDICTIONS_FILE,
    TRAIN_FRAME_FILE,
    TRAINING_FILE,
)
from .utils import ensure_dirs, save_json


@dataclass
class TrainingArtifacts:
    model_name: str
    feature_columns: List[str]
    metrics: Dict[str, Dict[str, float]]


def load_training_frame() -> pd.DataFrame:
    return pd.read_csv(TRAINING_FILE)


def get_feature_columns(frame: pd.DataFrame) -> List[str]:
    non_features = {"soc_code", "aioe_title", "aioe_target", "occupation_title", "description", "risk_band"}
    return [c for c in frame.columns if c not in non_features]


def train_models(requested_model: str = "rule_based") -> TrainingArtifacts:
    ensure_dirs([MODEL_DIR])
    frame = load_training_frame().copy()
    feature_columns = get_feature_columns(frame)

    metrics = {
        "rule_based": {
            "n_occupations": int(len(frame)),
            "score_min": float(frame["aei_score"].min()),
            "score_mean": float(frame["aei_score"].mean()),
            "score_max": float(frame["aei_score"].max()),
            "green_share": float((frame["risk_band"] == "Green").mean()),
            "yellow_share": float((frame["risk_band"] == "Yellow").mean()),
            "red_share": float((frame["risk_band"] == "Red").mean()),
        }
    }

    save_json({"feature_columns": feature_columns, "best_model": "rule_based"}, FEATURES_FILE)
    save_json(metrics, METRICS_FILE)

    prediction_frame = frame[["soc_code", "occupation_title", "aei_score", "risk_band"]].copy()
    prediction_frame = prediction_frame.rename(columns={"aei_score": "predicted_exposure"})
    prediction_frame.to_csv(PREDICTIONS_FILE, index=False)
    frame.to_csv(TRAIN_FRAME_FILE, index=False)

    return TrainingArtifacts(model_name="rule_based", feature_columns=feature_columns, metrics=metrics)


def predict_one(soc_code: str, top_k: int = 8) -> dict:
    frame = pd.read_csv(TRAIN_FRAME_FILE if TRAIN_FRAME_FILE.exists() else TRAINING_FILE)
    row = frame.loc[frame["soc_code"].astype(str) == str(soc_code)]
    if row.empty:
        raise KeyError(f"SOC code not found: {soc_code}")
    row = row.iloc[0]

    contrib = pd.read_csv(ABILITY_CONTRIBUTIONS_FILE)
    contrib_row = contrib.loc[contrib["soc_code"].astype(str) == str(soc_code)].copy()
    if contrib_row.empty:
        top = []
    else:
        contrib_row = contrib_row.sort_values("ability_contribution", ascending=False).head(top_k)
        top = contrib_row[[
            "ability_name", "ability_weight", "exposure_sigmoid", "ability_contribution"
        ]].round(4).rename(columns={
            "ability_name": "feature",
            "ability_weight": "weight",
            "exposure_sigmoid": "normalized_exposure",
            "ability_contribution": "contribution"
        }).to_dict(orient="records")

    return {
        "soc_code": str(row["soc_code"]),
        "occupation_title": row["occupation_title"],
        "aei_score": round(float(row["aei_score"]), 4),
        "risk_band": row["risk_band"],
        "legacy_aioe_target": None if pd.isna(row.get("aioe_target")) else round(float(row.get("aioe_target")), 4),
        "top_contributors": top,
    }
