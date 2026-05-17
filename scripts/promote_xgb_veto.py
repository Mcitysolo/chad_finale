#!/usr/bin/env python3
"""scripts/promote_xgb_veto.py — XGB veto model promotion CLI.

Phase 3 of the XGB Veto Model Artifact Hygiene plan
(docs/XGB_VETO_MODEL_ARTIFACT_HYGIENE_PLAN_2026-05-17.md).

The weekly trainer (chad-xgb-train.timer) lands candidate models under
``runtime/models/xgb_veto/candidates/<UTC ts>/``. This CLI is the
operator-controlled gate that decides whether a candidate becomes the
predictor's active model:

  python3 scripts/promote_xgb_veto.py --list
  python3 scripts/promote_xgb_veto.py --status
  python3 scripts/promote_xgb_veto.py --candidate <UTC_TIMESTAMP>
  python3 scripts/promote_xgb_veto.py --candidate <UTC_TIMESTAMP> \\
      --operator-approve "reason"

Auto-gate requires both ``accuracy_new >= accuracy_current`` and
``logloss_new <= logloss_current``. A regression on either metric
blocks promotion unless the operator passes ``--operator-approve``
with a reason; the reason is recorded in the promoted manifest so the
override is auditable.

The CLI is stdlib-only, makes no strategy / execution / risk imports,
never restarts services, and writes atomically.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

CANDIDATES_DIR = REPO_ROOT / "runtime" / "models" / "xgb_veto" / "candidates"
RUNTIME_MODEL_DIR = REPO_ROOT / "runtime" / "models" / "xgb_veto" / "current"
RUNTIME_MODEL_PATH = RUNTIME_MODEL_DIR / "xgb_veto_model.json"
RUNTIME_MANIFEST_PATH = RUNTIME_MODEL_DIR / "xgb_veto_manifest.json"

BASELINE_MODEL_PATH = REPO_ROOT / "shared" / "models" / "xgb_veto_model.json"
BASELINE_MANIFEST_PATH = REPO_ROOT / "shared" / "models" / "xgb_veto_manifest.json"

MODEL_FILENAME = "xgb_veto_model.json"
MANIFEST_FILENAME = "xgb_veto_manifest.json"


# ---------------------------------------------------------------------------
# Manifest IO helpers
# ---------------------------------------------------------------------------


def _read_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _metric(manifest: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not isinstance(manifest, dict):
        return None
    metrics = manifest.get("metrics") or {}
    if not isinstance(metrics, dict):
        return None
    raw = metrics.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _list_candidates() -> List[Tuple[str, Path]]:
    """Return (candidate_id, dir) pairs sorted newest first."""
    if not CANDIDATES_DIR.is_dir():
        return []
    pairs: List[Tuple[str, Path]] = []
    for child in CANDIDATES_DIR.iterdir():
        if not child.is_dir():
            continue
        manifest = child / MANIFEST_FILENAME
        model = child / MODEL_FILENAME
        if manifest.is_file() and model.is_file():
            pairs.append((child.name, child))
    pairs.sort(key=lambda p: p[0], reverse=True)
    return pairs


def _active_manifest() -> Tuple[Optional[Dict[str, Any]], str]:
    """Return (manifest, source). source is 'runtime' or 'baseline' or 'none'."""
    runtime = _read_manifest(RUNTIME_MANIFEST_PATH)
    if runtime is not None:
        return runtime, "runtime"
    baseline = _read_manifest(BASELINE_MANIFEST_PATH)
    if baseline is not None:
        return baseline, "baseline"
    return None, "none"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_list() -> int:
    pairs = _list_candidates()
    if not pairs:
        print(f"[promote_xgb_veto] no candidates under {CANDIDATES_DIR}")
        return 0
    print(f"[promote_xgb_veto] candidates under {CANDIDATES_DIR}:")
    for cid, cdir in pairs:
        manifest = _read_manifest(cdir / MANIFEST_FILENAME) or {}
        trained = manifest.get("trained_at_utc", "?")
        acc = _metric(manifest, "accuracy")
        loss = _metric(manifest, "logloss")
        acc_s = f"{acc:.4f}" if acc is not None else "?"
        loss_s = f"{loss:.4f}" if loss is not None else "?"
        print(f"  {cid}  trained={trained}  accuracy={acc_s}  logloss={loss_s}")
    return 0


def cmd_status() -> int:
    manifest, source = _active_manifest()
    if manifest is None:
        print("[promote_xgb_veto] no active model found "
              f"(runtime={RUNTIME_MANIFEST_PATH}, baseline={BASELINE_MANIFEST_PATH})")
        return 0
    version = manifest.get("model_version", "?")
    trained = manifest.get("trained_at_utc", "?")
    acc = _metric(manifest, "accuracy")
    loss = _metric(manifest, "logloss")
    acc_s = f"{acc:.4f}" if acc is not None else "?"
    loss_s = f"{loss:.4f}" if loss is not None else "?"
    print(f"[promote_xgb_veto] active model (source={source})")
    print(f"  model_version : {version}")
    print(f"  trained_at_utc: {trained}")
    print(f"  accuracy      : {acc_s}")
    print(f"  logloss       : {loss_s}")
    return 0


def _atomic_copy(src: Path, dst: Path) -> None:
    """Atomic copy: write to <dst>.tmp in the same dir, then os.replace."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_bytes(src.read_bytes())
    os.replace(str(tmp), str(dst))


def _atomic_write_json(payload: Dict[str, Any], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(str(tmp), str(dst))


def cmd_promote(candidate_id: str, operator_approve: Optional[str]) -> int:
    candidate_dir = CANDIDATES_DIR / candidate_id
    cand_model = candidate_dir / MODEL_FILENAME
    cand_manifest_path = candidate_dir / MANIFEST_FILENAME

    if not cand_model.is_file() or not cand_manifest_path.is_file():
        print(f"[promote_xgb_veto] candidate not found: {candidate_dir}",
              file=sys.stderr)
        return 1

    cand_manifest = _read_manifest(cand_manifest_path)
    if cand_manifest is None:
        print(f"[promote_xgb_veto] candidate manifest invalid: {cand_manifest_path}",
              file=sys.stderr)
        return 1

    current_manifest, source = _active_manifest()

    new_acc = _metric(cand_manifest, "accuracy")
    new_loss = _metric(cand_manifest, "logloss")
    cur_acc = _metric(current_manifest, "accuracy")
    cur_loss = _metric(current_manifest, "logloss")

    if new_acc is None or new_loss is None:
        print("[promote_xgb_veto] candidate metrics missing accuracy/logloss",
              file=sys.stderr)
        return 1

    # When no current active model exists yet, treat metric delta as 0
    # so the first-ever promotion does not require operator approval.
    delta_acc = new_acc - cur_acc if cur_acc is not None else 0.0
    delta_loss = new_loss - cur_loss if cur_loss is not None else 0.0

    gate_pass = (delta_acc >= 0.0) and (delta_loss <= 0.0)
    if not gate_pass and not operator_approve:
        print(
            "[promote_xgb_veto] promotion blocked — metric regression\n"
            f"  candidate accuracy={new_acc:.4f}  current={cur_acc}\n"
            f"  candidate logloss ={new_loss:.4f}  current={cur_loss}\n"
            f"  delta_accuracy={delta_acc:+.4f}  delta_logloss={delta_loss:+.4f}\n"
            "  re-run with --operator-approve \"<reason>\" to override.",
            file=sys.stderr,
        )
        return 1

    promoted_by = (
        f"operator_approve:{operator_approve}"
        if operator_approve
        else "auto_gate"
    )
    promoted_at = datetime.now(timezone.utc).isoformat()

    # Copy the candidate model atomically into RUNTIME_MODEL_PATH.
    try:
        _atomic_copy(cand_model, RUNTIME_MODEL_PATH)
    except Exception as exc:
        print(f"[promote_xgb_veto] failed to copy model: {exc}",
              file=sys.stderr)
        return 1

    # Promoted manifest carries the candidate's body plus a promotion
    # audit trail so operators can grep the active manifest to learn
    # who promoted it, when, from which candidate, and with what
    # metric delta. ``promoted_by`` distinguishes auto vs operator.
    promoted_manifest = dict(cand_manifest)
    promoted_manifest["promoted_at_utc"] = promoted_at
    promoted_manifest["promoted_from_candidate"] = candidate_id
    promoted_manifest["promoted_by"] = promoted_by
    promoted_manifest["promoted_from_source"] = source
    promoted_manifest["metrics_delta_accuracy"] = float(delta_acc)
    promoted_manifest["metrics_delta_logloss"] = float(delta_loss)
    promoted_manifest["model_path"] = str(RUNTIME_MODEL_PATH)
    if operator_approve:
        promoted_manifest["operator_approve_reason"] = operator_approve

    try:
        _atomic_write_json(promoted_manifest, RUNTIME_MANIFEST_PATH)
    except Exception as exc:
        print(f"[promote_xgb_veto] failed to write manifest: {exc}",
              file=sys.stderr)
        return 1

    print(f"[promote_xgb_veto] promoted candidate {candidate_id} -> current")
    print(f"  promoted_by   : {promoted_by}")
    print(f"  delta_accuracy: {delta_acc:+.4f}")
    print(f"  delta_logloss : {delta_loss:+.4f}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote an XGB veto candidate to the active runtime model.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list", action="store_true",
        help="List available candidates (newest first).",
    )
    group.add_argument(
        "--status", action="store_true",
        help="Report the currently active model (runtime or baseline).",
    )
    group.add_argument(
        "--candidate", metavar="UTC_TIMESTAMP",
        help="Candidate id (timestamp dir name) to promote.",
    )
    parser.add_argument(
        "--operator-approve", metavar="REASON",
        help="Override the metric gate; reason recorded in promoted manifest.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.operator_approve and not args.candidate:
        print("[promote_xgb_veto] --operator-approve requires --candidate",
              file=sys.stderr)
        return 1

    try:
        if args.list:
            return cmd_list()
        if args.status:
            return cmd_status()
        if args.candidate:
            return cmd_promote(args.candidate, args.operator_approve)
    except Exception as exc:  # noqa: BLE001 — CLI surface
        print(f"[promote_xgb_veto] unexpected error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
