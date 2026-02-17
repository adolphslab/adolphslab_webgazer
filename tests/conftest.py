from __future__ import annotations

from pathlib import Path

import sys

# Allow running tests without installing the package (src/ layout).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_tables(tmp_path: Path) -> tuple[Path, Path]:
    """Create tiny synthetic df_mergedmain.csv and df_runs.csv."""
    rng = np.random.default_rng(123)

    groups = ["Control", "ASD", "SPARK"]
    sexes = ["Male", "Female"]
    videos = ["S1", "S2"]
    task = "task5_pro"

    feat_rows = []
    sid = 0
    for group in groups:
        for i in range(6):
            sid += 1
            sex = sexes[i % 2]
            for video in videos:
                for feat in ["speaker", "distraction_offspeaker", "speaker_not_averted", "speaker_averted"]:
                    trials = int(rng.integers(50, 80))
                    prob = 0.6
                    if feat == "speaker" and group != "Control":
                        prob = 0.5
                    if feat == "distraction_offspeaker" and group != "Control":
                        prob = 0.2
                    if feat.startswith("speaker_"):
                        prob = 0.55
                    succ = int(rng.binomial(trials, prob))
                    feat_rows.append(
                        {
                            "SID": f"S{sid:03d}",
                            "Video": video,
                            "task": task,
                            "feature": feat,
                            "Group": group,
                            "Sex": sex,
                            "use_run": True,
                            "dwell_bins": succ,
                            "n_present_any_valid": trials,
                            "n_y_valid": trials,
                        }
                    )

    df_feat = pd.DataFrame(feat_rows)

    run_rows = []
    sid = 0
    for group in groups:
        for i in range(6):
            sid += 1
            sex = sexes[i % 2]
            for video in videos:
                trials = int(rng.integers(80, 120))
                prob = 0.05 if group == "Control" else 0.07
                succ = int(rng.binomial(trials, prob))
                run_rows.append(
                    {
                        "SID": f"S{sid:03d}",
                        "Video": video,
                        "task": task,
                        "Group": group,
                        "Sex": sex,
                        "use_run": True,
                        "n_switches": succ,
                        "n_contig_steps": trials,
                    }
                )
    df_runs = pd.DataFrame(run_rows)

    feat_path = tmp_path / "df_mergedmain.csv"
    runs_path = tmp_path / "df_runs.csv"
    df_feat.to_csv(feat_path, index=False)
    df_runs.to_csv(runs_path, index=False)
    return feat_path, runs_path
