from __future__ import annotations

from alabwebgazer.core.provenance import safe_env_allowlist


def test_safe_env_allowlist_redacts_sensitive_prefixed_values(monkeypatch) -> None:
    monkeypatch.setenv("ALABWEBGAZER_HASH_SALT", "super-secret-salt")
    monkeypatch.setenv("ALABWEBGAZER_PIPELINE_MODE", "demo")
    monkeypatch.setenv("UNRELATED_SECRET", "should-not-appear")

    env = safe_env_allowlist()
    assert env["ALABWEBGAZER_HASH_SALT"] == "<redacted>"
    assert env["ALABWEBGAZER_PIPELINE_MODE"] == "demo"
    assert "UNRELATED_SECRET" not in env
