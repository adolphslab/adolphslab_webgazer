"""Helpers for working with jsPsych/WebGazer exports.

These utilities are parsers/adapters; study-specific mappings (stimulus -> condition/length)
should live in configuration and pipeline code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


def load_jspsych_json(path: str | Path) -> list[dict[str, Any]]:
    """Load a jsPsych JSON export.

    Supports either:
    - list[dict] (typical export)
    - dict containing a list under one of: data, trials, raw, rows
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        obj = json.load(fh)

    if isinstance(obj, list):
        return [dict(x) for x in obj]
    if isinstance(obj, dict):
        for key in ("data", "trials", "raw", "rows"):
            val = obj.get(key)
            if isinstance(val, list):
                return [dict(x) for x in val]

    raise ValueError(f"Unsupported jsPsych JSON shape in {p}")


def find_trials(data: Iterable[Mapping[str, Any]], trial_type: str) -> list[int]:
    """Return indices of trials whose ``trial_type`` matches exactly."""
    idx: list[int] = []
    for i, row in enumerate(data):
        if str(row.get("trial_type", "")) == trial_type:
            idx.append(i)
    return idx


def trial_webgazer_frame(trial: Mapping[str, Any]) -> pd.DataFrame:
    """Extract the raw WebGazer stream from a jsPsych trial."""
    raw = trial.get("webgazer_data")
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        raise ValueError("trial has no 'webgazer_data' list")
    df = pd.DataFrame(raw)
    missing = sorted({"t", "x", "y"} - set(df.columns))
    if missing:
        raise ValueError(f"webgazer_data missing required keys: {missing}")
    return df[["t", "x", "y"]].copy()


def trial_screen_dims(trial: Mapping[str, Any]) -> tuple[float, float]:
    """Best-effort extraction of innerWidth/innerHeight (pixels)."""
    for w_key, h_key in (("innerWidth", "innerHeight"), ("windowWidth", "windowHeight")):
        w = trial.get(w_key)
        h = trial.get(h_key)
        if isinstance(w, (int, float)) and isinstance(h, (int, float)):
            return float(w), float(h)
    raise ValueError("Could not find screen dimensions (innerWidth/innerHeight) in trial")


def coerce_stimulus_text(trial: Mapping[str, Any]) -> str:
    """Return a best-effort string representation of the stimulus field."""
    stim = trial.get("stimulus", "")
    if isinstance(stim, list):
        return " ".join([str(x) for x in stim])
    return str(stim)


def require_keys(trial: Mapping[str, Any], keys: Sequence[str]) -> None:
    missing = [k for k in keys if k not in trial]
    if missing:
        raise ValueError(f"trial missing keys: {missing}")
