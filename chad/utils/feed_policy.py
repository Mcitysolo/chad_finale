"""
chad/utils/feed_policy.py — W4A-8 (DQ1/DQ3): per-feed dependency classes +
prewritten stale policies.

`read_with_policy(path)` wraps runtime_json.read_runtime_state_json and returns
`(obj, FeedVerdict)`. The verdict carries the feed's declared class + policy +
the ACTION a reader should take (block_entries / degrade_loud / hold_last_loud /
ignore) given the freshness result. XOV2 semantics: a typed unknown is never a
value indistinguishable from a legitimate reading — a stale/corrupt/missing feed
yields obj possibly-None AND an explicit non-fresh verdict, never a silent
fall-through to a default that reads like real data.

Gated by CHAD_DQ_POLICIES (off|shadow|enforce, default off). Shadow computes the
verdict + writes evidence but the ACTION is advisory (a reader treats
`should_block` as False); enforce lets a reader honor block_entries. The wiring
at each site decides how to spend the verdict — this module only classifies.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from chad.risk.fuse_box import ENV_DQ, MODE_ENFORCE, MODE_OFF, fuse_mode

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "feed_policies.json"
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "data" / "fuse_box"

# Freshness outcomes.
FRESH = "fresh"
STALE = "stale"
CORRUPT = "corrupt"
MISSING = "missing"

# Policy verbs.
BLOCK_ENTRIES = "block_entries"
DEGRADE_LOUD = "degrade_loud"
HOLD_LAST_LOUD = "hold_last_loud"
IGNORE = "ignore"
_VALID_POLICIES = frozenset({BLOCK_ENTRIES, DEGRADE_LOUD, HOLD_LAST_LOUD, IGNORE})


@dataclasses.dataclass(frozen=True)
class FeedVerdict:
    feed: str
    feed_class: str
    freshness: str        # fresh | stale | corrupt | missing
    policy: str           # the verb that applies to THIS freshness
    action: str           # what a reader should DO (== policy, or "ignore" in shadow)
    mode: str             # off | shadow | enforce
    age_seconds: Optional[float]
    ttl_seconds: Optional[float]
    reason: str

    @property
    def should_block(self) -> bool:
        """True only when the reader must refuse NEW entries: policy is
        block_entries, the feed is non-fresh, AND DQ is enforcing. Shadow and
        off never block (margin-gate contract)."""
        return (
            self.action == BLOCK_ENTRIES
            and self.freshness != FRESH
            and self.mode == MODE_ENFORCE
        )


class FeedPolicies:
    """Parsed config/feed_policies.json. Missing/corrupt → empty (every feed
    then classifies as advisory/ignore — a broken policy file can never
    tighten behavior, only disarm the DQ layer)."""

    def __init__(self, feeds: Mapping[str, Mapping[str, Any]]):
        self._feeds = dict(feeds)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "FeedPolicies":
        target = Path(path) if path is not None else CONFIG_PATH
        try:
            obj = json.loads(target.read_text(encoding="utf-8"))
            feeds = obj.get("feeds") if isinstance(obj, dict) else None
            if not isinstance(feeds, dict):
                raise ValueError("no feeds object")
        except Exception as exc:  # noqa: BLE001
            LOG.warning("FEED_POLICIES_UNREADABLE path=%s err=%s", target, exc)
            feeds = {}
        return cls(feeds)

    def for_feed(self, rel_path: str) -> Optional[Mapping[str, Any]]:
        return self._feeds.get(rel_path)


def _policy_for(cfg: Optional[Mapping[str, Any]], freshness: str) -> Tuple[str, str]:
    """(policy_verb, feed_class) for a freshness outcome. Unknown feed →
    (ignore, advisory). A policy value not in the vocabulary → ignore
    (fail-safe: a typo can never block trading)."""
    if not isinstance(cfg, Mapping):
        return IGNORE, "advisory"
    feed_class = str(cfg.get("class") or "advisory")
    if freshness == FRESH:
        return IGNORE, feed_class
    key = {
        STALE: "on_stale", CORRUPT: "on_corrupt", MISSING: "on_missing",
    }.get(freshness, "on_stale")
    verb = str(cfg.get(key) or IGNORE).strip().lower()
    if verb not in _VALID_POLICIES:
        LOG.warning("FEED_POLICY_UNKNOWN_VERB feed_class=%s verb=%s → ignore",
                    feed_class, verb)
        verb = IGNORE
    return verb, feed_class


def classify_freshness(
    obj: Optional[Dict[str, Any]], freshness_obj: Any, path: Path
) -> str:
    """Map a runtime_json read result to fresh/stale/corrupt/missing. A file
    that does not exist → missing; a file that exists but doesn't parse to an
    object → corrupt; an object past its ttl → stale."""
    if not path.exists():
        return MISSING
    if obj is None:
        # Existed but unreadable/non-object → corrupt (distinct from missing).
        return CORRUPT
    return FRESH if getattr(freshness_obj, "ok", False) else STALE


def read_with_policy(
    rel_or_abs_path: Any,
    *,
    policies: Optional[FeedPolicies] = None,
    env: Optional[Mapping[str, str]] = None,
    repo_root: Optional[Path] = None,
    evidence_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], FeedVerdict]:
    """Read a runtime state file and classify it against its declared policy.

    Accepts a repo-relative key ("runtime/scr_state.json") or an absolute
    path; the policy lookup keys on the relative form. Returns (obj, verdict);
    obj may be non-None yet stale (the caller decides via verdict) — obj is
    NEVER a fabricated default (XOV2: a typed unknown, never a value that
    reads like a legitimate reading)."""
    from chad.utils.runtime_json import read_runtime_state_json

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    p = Path(rel_or_abs_path)
    rel = str(p) if not p.is_absolute() else _rel_to_root(p, root)
    abs_path = p if p.is_absolute() else (root / p)

    pols = policies or FeedPolicies.load()
    cfg = pols.for_feed(rel)
    mode = fuse_mode(ENV_DQ, env)

    obj, freshness = read_runtime_state_json(abs_path)
    fresh_state = classify_freshness(obj, freshness, abs_path)
    policy, feed_class = _policy_for(cfg, fresh_state)
    # In shadow/off the ACTION never blocks; in enforce it equals the policy.
    action = policy if mode == MODE_ENFORCE else (
        policy if policy in (DEGRADE_LOUD, HOLD_LAST_LOUD, IGNORE) else IGNORE
    )

    verdict = FeedVerdict(
        feed=rel,
        feed_class=feed_class,
        freshness=fresh_state,
        policy=policy,
        action=action,
        mode=mode,
        age_seconds=getattr(freshness, "age_seconds", None),
        ttl_seconds=getattr(freshness, "ttl_seconds", None),
        reason=getattr(freshness, "reason", fresh_state),
    )
    if mode != MODE_OFF and fresh_state != FRESH and policy != IGNORE:
        _emit_dq(verdict, evidence_dir=evidence_dir)
    return obj, verdict


def _rel_to_root(abs_path: Path, root: Path) -> str:
    try:
        return str(abs_path.relative_to(root))
    except Exception:  # noqa: BLE001
        return str(abs_path)


def _emit_dq(verdict: FeedVerdict, evidence_dir: Optional[Path] = None) -> None:
    """Loud surface for a non-fresh feed: DQ marker + evidence row + coach
    feed_stale alert (value-free dedupe). Never raises past the pytest guard."""
    try:
        marker = (
            "DQ_INPUT_DEAD" if verdict.action == BLOCK_ENTRIES
            else "DQ_FEED_DEGRADED"
        )
        LOG.warning(
            "%s feed=%s class=%s freshness=%s policy=%s action=%s mode=%s age=%s",
            marker, verdict.feed, verdict.feed_class, verdict.freshness,
            verdict.policy, verdict.action, verdict.mode, verdict.age_seconds,
        )
        from chad.risk.fuse_box import append_evidence

        append_evidence(
            [
                {
                    "event": "dq_feed",
                    "marker": marker,
                    "feed": verdict.feed,
                    "class": verdict.feed_class,
                    "freshness": verdict.freshness,
                    "policy": verdict.policy,
                    "action": verdict.action,
                    "mode": verdict.mode,
                }
            ],
            evidence_dir=evidence_dir,
        )
        try:
            from chad.utils.coach_voice import format_alert
            from chad.utils.telegram_notify import notify

            facts = {
                "title": f"Data feed {verdict.freshness}: {verdict.feed}",
                "summary": (
                    f"{verdict.feed} is {verdict.freshness}. Policy "
                    f"'{verdict.policy}' "
                    + ("is refusing new entries (exits stay free)."
                       if verdict.should_block else
                       "logged it; no behavior change beyond this alert.")
                ),
            }
            msg = format_alert("feed_stale", facts)
            notify(
                msg if isinstance(msg, str) and msg.strip() else facts["summary"],
                severity="critical" if verdict.should_block else "warning",
                dedupe_key=_dq_dedupe_key(verdict),
                raise_on_fail=False,
            )
        except Exception:  # noqa: BLE001 — NOTIFY best-effort
            pass
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        LOG.warning("DQ_EMIT_FAILED feed=%s err=%s", verdict.feed, exc)


def _dq_dedupe_key(verdict: FeedVerdict) -> str:
    import re

    ident = re.sub(r"\d+", "", verdict.feed.strip().lower())
    return f"dq_{verdict.freshness}_{ident}"


__all__ = [
    "BLOCK_ENTRIES",
    "CORRUPT",
    "DEGRADE_LOUD",
    "FRESH",
    "HOLD_LAST_LOUD",
    "IGNORE",
    "MISSING",
    "STALE",
    "FeedPolicies",
    "FeedVerdict",
    "classify_freshness",
    "read_with_policy",
]
