from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alabwebgazer.core.config import ConfigError, load_config


def test_unified_config_rejects_duplicate_steps(tmp_path: Path) -> None:
    cfg = {
        "run": {"output_root": str(tmp_path / "runs")},
        "pipeline": {
            "steps": [
                "preprocess_jspsych",
                "build_tables",
                "build_tables",
            ]
        },
        "preprocess_jspsych": {"enabled": True, "config_path": "pre.yml"},
        "build_tables": {"enabled": True, "config_path": "bt.yml"},
        "analysis_ready": {"enabled": True, "config_path": "ar.yml"},
    }
    cfg_path = tmp_path / "full.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError):
        _ = load_config(cfg_path)
