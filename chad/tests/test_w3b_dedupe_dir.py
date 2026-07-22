"""W3B-10 — telegram dedupe state relocated to runtime/dedupe/ + policy.

Locked properties:
- _dedupe_path lands under runtime/dedupe/ with the SAME filename derivation
  (the e2e naming pins in test_health_alert_pipeline_e2e assert only names —
  they keep passing untouched);
- _dedupe_mark creates the directory on first write;
- the BOX-042 cleanup tool scans BOTH the legacy loose location and the new
  subdirectory (migration window);
- repo-side scheduled-cleanup units + install doc exist (BOX-042 left
  scheduling manual; nothing installed by this commit).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# path derivation
# ---------------------------------------------------------------------------


def test_dedupe_path_is_under_runtime_dedupe():
    from chad.utils.telegram_notify import RUNTIME_DIR, _dedupe_path

    p = _dedupe_path("health_R19_IBKRsustainedlatency")
    assert p.parent == RUNTIME_DIR / "dedupe"
    # filename derivation unchanged (the e2e pins depend on it)
    assert p.name == "telegram_dedupe_health_R19_IBKRsustainedlatency.json"


def test_dedupe_mark_creates_directory(tmp_path, monkeypatch):
    from chad.utils import telegram_notify as tn

    monkeypatch.setattr(tn, "RUNTIME_DIR", tmp_path)
    tn._dedupe_mark("w3b_test_key")
    p = tmp_path / "dedupe" / "telegram_dedupe_w3b_test_key.json"
    assert p.is_file()
    assert "last_sent_unix" in json.loads(p.read_text(encoding="utf-8"))
    # and _dedupe_allows reads it back (suppression active within TTL)
    cfg = type("C", (), {"dedupe_ttl_s": 900})()
    assert tn._dedupe_allows(cfg, "w3b_test_key") is False


def test_dedupe_allows_when_no_file(tmp_path, monkeypatch):
    from chad.utils import telegram_notify as tn

    monkeypatch.setattr(tn, "RUNTIME_DIR", tmp_path)
    cfg = type("C", (), {"dedupe_ttl_s": 900})()
    assert tn._dedupe_allows(cfg, "never_sent") is True


# ---------------------------------------------------------------------------
# cleanup tool migration window
# ---------------------------------------------------------------------------


def _write_stale(path: Path, age_s: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"last_sent_unix": time.time() - age_s}), encoding="utf-8")
    old = time.time() - age_s
    import os

    os.utime(path, (old, old))


def test_cleanup_tool_scans_both_locations(tmp_path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "cleanup_telegram_dedupe", REPO_ROOT / "ops" / "cleanup_telegram_dedupe.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cleanup_telegram_dedupe"] = mod  # dataclass needs the registration
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("cleanup_telegram_dedupe", None)

    legacy = tmp_path / "telegram_dedupe_legacy_key.json"
    nested = tmp_path / "dedupe" / "telegram_dedupe_new_key.json"
    _write_stale(legacy, 10 * 86400)
    _write_stale(nested, 10 * 86400)
    (tmp_path / "unrelated.json").write_text("{}", encoding="utf-8")

    found = {p.name for p in mod._iter_dedupe_files(tmp_path)}
    assert found == {
        "telegram_dedupe_legacy_key.json",
        "telegram_dedupe_new_key.json",
    }


# ---------------------------------------------------------------------------
# policy artifacts (repo-side only)
# ---------------------------------------------------------------------------


def test_cleanup_units_and_install_doc_exist():
    service = REPO_ROOT / "ops" / "systemd" / "chad-dedupe-cleanup.service"
    timer = REPO_ROOT / "ops" / "systemd" / "chad-dedupe-cleanup.timer"
    doc = REPO_ROOT / "docs" / "DEDUPE_CLEANUP_INSTALL.md"
    assert service.is_file() and timer.is_file() and doc.is_file()
    s = service.read_text(encoding="utf-8")
    assert "OnFailure=chad-service-alert@%N.service" in s  # P0A-A2 parity
    assert "cleanup_telegram_dedupe.py --apply" in s
    t = timer.read_text(encoding="utf-8")
    assert "OnCalendar=weekly" in t and "Persistent=true" in t
    d = doc.read_text(encoding="utf-8")
    assert "systemctl daemon-reload" in d
    assert "split-brain" in d.lower()
