from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import pandas as pd
import requests

from .config import (
    ABILITIES_FILE,
    ABILITY_CONTRIBUTIONS_FILE,
    AIOE_FILE,
    AIOE_URL,
    DATA_DIR,
    OCCUPATION_LOOKUP_FILE,
    OCCUPATIONS_FILE,
    ONET_ABILITIES_URL,
    ONET_OCCUPATIONS_URL,
    ONET_TASKS_URL,
    ONET_TASK_RATINGS_URL,
    ONET_TASKS_TO_DWAS_URL,
    TASKS_FILE,
    TASK_RATINGS_FILE,
    TASKS_TO_DWAS_FILE,
    TRAINING_FILE,
)
from .utils import ensure_dirs

TASK_AUTOMATION_FILE = DATA_DIR / "task_automation.csv"


def fetch_raw_data(timeout: int = 60) -> Tuple[str, str, str]:
    ensure_dirs([DATA_DIR])
    for url, path in [
        (ONET_ABILITIES_URL, ABILITIES_FILE),
        (ONET_OCCUPATIONS_URL, OCCUPATIONS_FILE),
        (ONET_TASKS_URL, TASKS_FILE),
        (ONET_TASK_RATINGS_URL, TASK_RATINGS_FILE),
        (ONET_TASKS_TO_DWAS_URL, TASKS_TO_DWAS_FILE),
        (AIOE_URL, AIOE_FILE),
    ]:
        if path.exists():
            continue
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        path.write_bytes(response.content)
    return str(ABILITIES_FILE), str(OCCUPATIONS_FILE), str(AIOE_FILE)


def _normalize_series(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    min_v, max_v = series.min(), series.max()
    if pd.isna(min_v) or pd.isna(max_v) or min_v == max_v:
        return pd.Series(np.full(len(series), 0.5), index=series.index)
    return (series - min_v) / (max_v - min_v)


def _sigmoid(series: pd.Series) -> pd.Series:
    arr = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return pd.Series(1.0 / (1.0 + np.exp(-arr)), index=series.index)


def _risk_band(score: float) -> str:
    if score < 0.33:
        return "Green"
    if score <= 0.66:
        return "Yellow"
    return "Red"


def build_training_dataset() -> pd.DataFrame:
    abilities = pd.read_csv(ABILITIES_FILE, sep="	")
    occupations = pd.read_csv(OCCUPATIONS_FILE, sep="	")
    aioe_occ = pd.read_excel(AIOE_FILE, sheet_name="Appendix A")
    aioe_ability = pd.read_excel(AIOE_FILE, sheet_name="Appendix E")

    abilities = abilities.rename(columns={
        "O*NET-SOC Code": "onet_soc_code",
        "Element Name": "ability_name",
        "Scale ID": "scale_id",
        "Data Value": "data_value",
    })
    abilities["soc_code"] = abilities["onet_soc_code"].astype(str).str.slice(0, 7)
    abilities["ability_name"] = abilities["ability_name"].astype(str).str.strip()
    abilities["scale_id"] = abilities["scale_id"].astype(str).str.strip()
    abilities["data_value"] = pd.to_numeric(abilities["data_value"], errors="coerce")
    abilities = abilities.dropna(subset=["soc_code", "ability_name", "scale_id", "data_value"])

    aioe_occ = aioe_occ.rename(columns={"SOC Code": "soc_code", "Occupation Title": "aioe_title", "AIOE": "aioe_target"})
    aioe_occ["soc_code"] = aioe_occ["soc_code"].astype(str).str.strip()
    aioe_occ["aioe_target"] = pd.to_numeric(aioe_occ["aioe_target"], errors="coerce")

    aioe_ability = aioe_ability.rename(columns={
        "O*NET Abilities": "ability_name",
        "Ability-Level AI Exposure": "ability_exposure"
    })
    aioe_ability["ability_name"] = aioe_ability["ability_name"].astype(str).str.strip()
    aioe_ability["ability_exposure"] = pd.to_numeric(aioe_ability["ability_exposure"], errors="coerce")

    im = abilities[abilities["scale_id"] == "IM"].copy()
    lv = abilities[abilities["scale_id"] == "LV"].copy()

    im = im.rename(columns={"data_value": "importance_raw"})[["soc_code", "ability_name", "importance_raw"]]
    lv = lv.rename(columns={"data_value": "level_raw"})[["soc_code", "ability_name", "level_raw"]]

    contrib = im.merge(lv, on=["soc_code", "ability_name"], how="left")
    contrib = contrib.merge(aioe_ability[["ability_name", "ability_exposure"]], on="ability_name", how="left")
    contrib["ability_exposure"] = contrib["ability_exposure"].fillna(0.0)
    contrib["level_raw"] = contrib["level_raw"].fillna(contrib["level_raw"].median())

    contrib["importance_norm_global"] = _normalize_series(contrib["importance_raw"])
    contrib["level_norm_global"] = _normalize_series(contrib["level_raw"])
    contrib["importance_level_blend"] = 0.7 * contrib["importance_norm_global"] + 0.3 * contrib["level_norm_global"]
    contrib["exposure_sigmoid"] = _sigmoid(contrib["ability_exposure"])

    weight_sum = contrib.groupby("soc_code")["importance_level_blend"].transform("sum")
    contrib["ability_weight"] = np.where(weight_sum > 0, contrib["importance_level_blend"] / weight_sum, 0.0)
    contrib["ability_contribution"] = contrib["ability_weight"] * contrib["exposure_sigmoid"]

    occ_scores = contrib.groupby("soc_code").agg(
        raw_aei_score=("ability_contribution", "sum"),
        mean_importance=("importance_norm_global", "mean"),
        std_importance=("importance_norm_global", "std"),
        high_importance_count=("importance_norm_global", lambda s: int((s >= 0.60).sum())),
        mean_level=("level_norm_global", "mean"),
        weighted_exposure_index=("ability_contribution", "sum"),
    ).reset_index()
    occ_scores["std_importance"] = occ_scores["std_importance"].fillna(0.0)
    occ_scores["aei_score"] = _normalize_series(occ_scores["raw_aei_score"])
    occ_scores["risk_band"] = occ_scores["aei_score"].apply(_risk_band)

    occupation_lookup = occupations.rename(columns={
        "O*NET-SOC Code": "onet_soc_code",
        "Title": "occupation_title",
        "Description": "description",
    }).copy()
    occupation_lookup["soc_code"] = occupation_lookup["onet_soc_code"].astype(str).str.slice(0, 7)
    occupation_lookup = occupation_lookup[["soc_code", "occupation_title", "description"]].drop_duplicates("soc_code")

    merged = occ_scores.merge(aioe_occ, on="soc_code", how="inner")
    merged = merged.merge(occupation_lookup, on="soc_code", how="left")
    merged["occupation_title"] = merged["occupation_title"].fillna(merged["aioe_title"])

    contrib = contrib.merge(merged[["soc_code", "occupation_title", "aei_score", "risk_band"]], on="soc_code", how="inner")
    contrib = contrib[[
        "soc_code", "occupation_title", "ability_name", "importance_raw", "level_raw",
        "importance_norm_global", "level_norm_global", "ability_exposure", "exposure_sigmoid",
        "ability_weight", "ability_contribution", "aei_score", "risk_band"
    ]].sort_values(["soc_code", "ability_contribution"], ascending=[True, False])

    occupation_lookup_final = merged[["soc_code", "occupation_title", "aioe_title", "description", "aei_score", "risk_band"]].drop_duplicates("soc_code")
    occupation_lookup_final.to_csv(OCCUPATION_LOOKUP_FILE, index=False)
    contrib.to_csv(ABILITY_CONTRIBUTIONS_FILE, index=False)
    merged.to_csv(TRAINING_FILE, index=False)

    # Build task-level automation estimates
    _build_task_automation(contrib)

    return merged


def _build_task_automation(ability_contrib: pd.DataFrame) -> pd.DataFrame:
    """Estimate task-level automation potential using O*NET tasks + ability exposure.

    For each task in an occupation we compute an automation score based on the
    occupation's overall ability-exposure profile and the task's importance.
    Tasks that are more routine / lower importance tend to be more automatable.
    """
    if not TASKS_FILE.exists() or not TASK_RATINGS_FILE.exists():
        return pd.DataFrame()

    # Load task statements
    tasks = pd.read_csv(TASKS_FILE, sep="\t")
    tasks = tasks.rename(columns={
        "O*NET-SOC Code": "onet_soc_code",
        "Task ID": "task_id",
        "Task": "task_description",
        "Task Type": "task_type",
    })
    tasks["soc_code"] = tasks["onet_soc_code"].astype(str).str.slice(0, 7)
    tasks = tasks[tasks["task_type"] == "Core"]  # Focus on core tasks

    # Load task importance ratings (Scale ID = "IM")
    ratings = pd.read_csv(TASK_RATINGS_FILE, sep="\t")
    ratings = ratings.rename(columns={
        "O*NET-SOC Code": "onet_soc_code",
        "Task ID": "task_id",
        "Scale ID": "scale_id",
        "Data Value": "task_importance",
    })
    ratings = ratings[ratings["scale_id"] == "IM"]
    ratings["soc_code"] = ratings["onet_soc_code"].astype(str).str.slice(0, 7)
    ratings["task_importance"] = pd.to_numeric(ratings["task_importance"], errors="coerce")

    # Merge tasks with importance
    task_df = tasks[["soc_code", "task_id", "task_description"]].merge(
        ratings[["soc_code", "task_id", "task_importance"]],
        on=["soc_code", "task_id"],
        how="inner",
    )

    # Get per-occupation mean AI exposure from abilities
    occ_exposure = ability_contrib.groupby("soc_code").agg(
        mean_exposure=("exposure_sigmoid", "mean"),
        weighted_exposure=("ability_contribution", "sum"),
    ).reset_index()

    task_df = task_df.merge(occ_exposure, on="soc_code", how="inner")

    # Normalize task importance within each occupation (0-1)
    task_df["importance_norm"] = task_df.groupby("soc_code")["task_importance"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else 0.5
    )

    # Estimate automation potential per task:
    # High-importance tasks in high-exposure occupations → highly automatable
    # We weight by the occupation's overall AI exposure profile
    task_df["automation_score"] = task_df["mean_exposure"] * (0.5 + 0.5 * task_df["importance_norm"])

    # Normalize across all tasks to 0-1
    task_df["automation_score"] = _normalize_series(task_df["automation_score"])

    # Assign automation level
    task_df["automation_level"] = task_df["automation_score"].apply(
        lambda x: "High" if x > 0.66 else ("Moderate" if x > 0.33 else "Low")
    )

    task_df = task_df[[
        "soc_code", "task_id", "task_description", "task_importance",
        "importance_norm", "automation_score", "automation_level",
    ]].sort_values(["soc_code", "automation_score"], ascending=[True, False])

    task_df.to_csv(TASK_AUTOMATION_FILE, index=False)
    return task_df
