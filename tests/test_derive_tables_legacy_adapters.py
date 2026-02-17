from __future__ import annotations

from pathlib import Path

import pytest

from alabwebgazer.derive_tables.config import DwellOptionsConfig, DwellTaskConfig
from alabwebgazer.derive_tables.legacy_adapters import _legacy_dwell_tag, run_dwell_task


def _task_cfg() -> DwellTaskConfig:
    return DwellTaskConfig(
        task="task5_pro",
        window_data_dir=Path("window_data"),
        video_feats_dir=Path("video_feats"),
        window_glob="videoview*.csv",
    )


def _options_cfg() -> DwellOptionsConfig:
    return DwellOptionsConfig(
        presence_threshold=0.5,
        min_run=1,
        bin_seconds=0.5,
        include_joint_features=True,
        derived_require_speaker_present=True,
        derived_require_speaker_unique=False,
        task5_require_speaker_unique=True,
        include_speaker_not_averted_loose=False,
        boundary_mode="posthoc",
        boundary_tau=0.05,
        boundary_power=1.0,
        save_bin_level=False,
    )


def test_run_dwell_task_prefers_expected_output_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from alabwebgazer.legacy import compute_dwelltime as dwell

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "window_data").mkdir(parents=True, exist_ok=True)
    (cfg_dir / "video_feats").mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    def fake_main(run_cfg: object) -> None:
        tag = _legacy_dwell_tag(run_cfg)
        expected = out_dir / f"{run_cfg.task}_{tag}.csv"
        expected.write_text("SID,feature,dwell_bins,n_present_any_valid\n", encoding="utf-8")
        # Decoy output that should not be selected.
        (out_dir / f"{run_cfg.task}_zzz_decoy.csv").write_text(
            "SID,feature,dwell_bins,n_present_any_valid\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(dwell, "main", fake_main)

    out = run_dwell_task(
        config_dir=cfg_dir,
        task_cfg=_task_cfg(),
        dwell_options=_options_cfg(),
        random_seed=0,
        out_dir=out_dir,
    )

    assert out.name.endswith(".csv")
    assert out.name.startswith("task5_pro_dwell_times")
    assert out.name != "task5_pro_zzz_decoy.csv"


def test_run_dwell_task_raises_on_ambiguous_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from alabwebgazer.legacy import compute_dwelltime as dwell

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "window_data").mkdir(parents=True, exist_ok=True)
    (cfg_dir / "video_feats").mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    def fake_main(run_cfg: object) -> None:
        # Intentionally omit expected filename and write two ambiguous candidates.
        (out_dir / f"{run_cfg.task}_alt_a.csv").write_text(
            "SID,feature,dwell_bins,n_present_any_valid\n",
            encoding="utf-8",
        )
        (out_dir / f"{run_cfg.task}_alt_b.csv").write_text(
            "SID,feature,dwell_bins,n_present_any_valid\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(dwell, "main", fake_main)

    with pytest.raises(RuntimeError, match="Ambiguous dwell outputs"):
        run_dwell_task(
            config_dir=cfg_dir,
            task_cfg=_task_cfg(),
            dwell_options=_options_cfg(),
            random_seed=0,
            out_dir=out_dir,
        )
