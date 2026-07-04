"""chad/validation/report_writer.py — Phase 5 signed report artifact (SSOT Part 4 / §3.8).

Assembles the harness's single output artifact and signs it: a machine-readable
``edge_report_<ts>.json`` (content-hashed) plus a human-readable ``.md`` (SSOT
``docs/CHAD_EDGE_VALIDATION_HARNESS_DESIGN_v1.1.md`` Part 4). The report embeds, in
one place, every input to the verdict so the decision is fully auditable:

  * the Phase-0 **data-quality** section (SSOT Phase 0 — "every later report embeds a
    data-quality section"),
  * the Phase-4 **feature-parity map** (REPLAYABLE / NOT_REPLAYABLE per head, SSOT §V1),
  * **per-head + portfolio metrics** and their verdicts (Part 4),
  * the **frozen config** hash + contents (SSOT §3.2),
  * the **OOS access count** and seal (SSOT §3.1),
  * the **universe-provenance** upward-bias flag (SSOT §V3 / Part 5).

**Signature (SSOT §3.8 determinism).** :func:`sign_report` computes a SHA-256 over the
canonical JSON of the report *without* its signature field and embeds it; the same
inputs therefore yield a byte-identical artifact and a stable signature, and
:func:`verify_signature` recomputes it to detect any post-hoc edit. Every timestamp is
caller-supplied — this module never reads the wall clock — so determinism holds.

Isolation (SSOT §1.2 / §2): standard-library only — :mod:`hashlib`, :mod:`json`,
:mod:`pathlib`, :mod:`typing`. It writes ONLY the two artifact files under the
caller-supplied ``out_dir`` (never ``runtime/``, never ``ready_for_live``), and it
performs no other I/O. It imports no sibling harness module — it consumes already-\
serialised section dicts, keeping it a pure formatter with no scoring/verdict logic of
its own (that lives in the phases that produced those dicts).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final, Mapping, Optional, Sequence

__all__ = [
    "SCHEMA_VERSION",
    "SIGNATURE_ALGO",
    "UNIVERSE_PROVENANCE_NOTE",
    "build_report",
    "sign_report",
    "verify_signature",
    "render_markdown",
    "write_report",
    "report_basename",
]

SCHEMA_VERSION: Final[str] = "edge_report.v1"
SIGNATURE_ALGO: Final[str] = "sha256"

# SSOT §V3 / Part 5: the 52-symbol universe was selected with knowledge of history, so
# every verdict is flagged with this known UPWARD (survivorship/selection) bias.
UNIVERSE_PROVENANCE_NOTE: Final[str] = (
    "Universe provenance (SSOT §V3): the symbol universe was selected with knowledge of "
    "history (survivorship/selection bias). Every verdict here carries a known UPWARD "
    "bias and is conditional on that universe. Results are evidence for a human decision "
    "only — the harness never flips ready_for_live (SSOT Part 0)."
)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    """Canonical JSON used for the signature (sorted keys, compact, ASCII, no NaN)."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def report_basename(generated_at: str) -> str:
    """Deterministic artifact basename ``edge_report_<ts>`` from a timestamp.

    Colons / spaces / other filesystem-hostile characters in ``generated_at`` are
    replaced with ``-`` so the same timestamp always yields the same, portable name.
    """
    if not isinstance(generated_at, str) or not generated_at:
        raise ValueError("generated_at must be a non-empty str")
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in generated_at)
    return f"edge_report_{safe}"


def build_report(
    *,
    generated_at: str,
    stage: str,
    final_run: bool,
    code_commit: str,
    data_quality: Mapping[str, Any],
    parity_map: Sequence[Mapping[str, Any]],
    parity_table: str,
    heads: Sequence[Mapping[str, Any]],
    portfolio: Mapping[str, Any],
    frozen_config: Mapping[str, Any],
    oos: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    verdict_summary: Mapping[str, Any],
    extra_notes: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Assemble the full (unsigned) report dict from already-serialised sections.

    Every argument is a plain, JSON-serialisable section dict/list produced by the
    upstream phases — this function only arranges them and stamps provenance. Pass
    :func:`sign_report` over the result to seal it. Deterministic in its inputs.
    """
    if not isinstance(generated_at, str) or not generated_at:
        raise ValueError("generated_at must be a non-empty str")
    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be a non-empty str")
    if not isinstance(final_run, bool):
        raise ValueError(f"final_run must be a bool, got {final_run!r}")

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "stage": stage,
        "final_run": final_run,
        "code_commit": code_commit,
        "provenance": {
            "universe_bias": UNIVERSE_PROVENANCE_NOTE,
            "oos_discipline": (
                "OOS is hash-sealed; scored only on an explicit --final-run and logged "
                "immutably (SSOT §3.1). This artifact records the access count; > 1 ⇒ "
                "CONTAMINATED."
            ),
            "replay_reconstruction": (
                "REPLAYABLE heads are replayed by a harness-side decision function over "
                "reconstructable (daily-bar) inputs; the live strategy module is never "
                "imported (SSOT §1.2 isolation). Reconstruction fidelity is a separate, "
                "flagged concern, not asserted here."
            ),
            "notes": list(extra_notes) if extra_notes else [],
        },
        "thresholds": dict(thresholds),
        "config_frozen": dict(frozen_config),
        "oos": dict(oos),
        "data_quality": dict(data_quality),
        "parity_map": [dict(p) for p in parity_map],
        "parity_table": parity_table,
        "heads": [dict(h) for h in heads],
        "portfolio": dict(portfolio),
        "verdict_summary": dict(verdict_summary),
    }


def sign_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``report`` with a ``signature`` block (content SHA-256) added.

    The signature is computed over the canonical JSON of the report with any existing
    ``signature`` key removed, so re-signing is idempotent and the digest covers exactly
    the content. Deterministic: identical content → identical signature.
    """
    body = {k: v for k, v in report.items() if k != "signature"}
    canonical = _canonical_json(body)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    signed = dict(body)
    signed["signature"] = {
        "algo": SIGNATURE_ALGO,
        "content_sha256": digest,
        "canonical_bytes": len(canonical.encode("utf-8")),
    }
    return signed


def verify_signature(signed: Mapping[str, Any]) -> bool:
    """Recompute the content hash and confirm it matches the embedded ``signature``.

    Returns ``False`` for a missing/malformed signature or any content edit since
    signing (the tamper-evidence check). Never raises on a merely-unsigned report.
    """
    sig = signed.get("signature")
    if not isinstance(sig, Mapping):
        return False
    claimed = sig.get("content_sha256")
    if not isinstance(claimed, str) or not claimed:
        return False
    body = {k: v for k, v in signed.items() if k != "signature"}
    canonical = _canonical_json(body)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest == claimed


# --------------------------------------------------------------------------- #
# Human-readable Markdown rendering (deterministic, pure formatting).
# --------------------------------------------------------------------------- #
def _fmt(value: Any) -> str:
    """Compact, deterministic scalar rendering for the Markdown tables."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        return f"{value:.6g}"
    return str(value)


def render_markdown(signed: Mapping[str, Any]) -> str:
    """Render the signed report as a deterministic human-readable Markdown document.

    Pure formatting over the report dict — no recomputation. Sections mirror the JSON
    artifact: header, verdict summary, OOS discipline, per-head verdicts, portfolio,
    data-quality, parity map, frozen config, and the signature.
    """
    lines: list[str] = []
    ap = lines.append

    ap(f"# CHAD Edge-Validation Report ({signed.get('schema_version', '?')})")
    ap("")
    ap(f"- **generated_at:** {_fmt(signed.get('generated_at'))}")
    ap(f"- **stage:** {_fmt(signed.get('stage'))}")
    final_run = bool(signed.get("final_run"))
    ap(f"- **final_run:** {_fmt(final_run)}"
       + ("" if final_run else "  _(dry run — decoy OOS; NOT evidence)_"))
    ap(f"- **code_commit:** {_fmt(signed.get('code_commit'))}")
    ap("")

    provenance = signed.get("provenance", {})
    if isinstance(provenance, Mapping):
        ap("> " + str(provenance.get("universe_bias", "")))
        ap("")

    # --- Verdict summary --------------------------------------------------- #
    ap("## Verdict summary")
    ap("")
    vs = signed.get("verdict_summary", {})
    if isinstance(vs, Mapping):
        counts = vs.get("counts")
        if isinstance(counts, Mapping) and counts:
            ap("| verdict | heads |")
            ap("| --- | --- |")
            for k in sorted(counts):
                ap(f"| {k} | {_fmt(counts[k])} |")
            ap("")
        pv = vs.get("portfolio_verdict")
        if pv is not None:
            ap(f"- **portfolio verdict:** {_fmt(pv)}")
            ap("")

    # --- OOS discipline ---------------------------------------------------- #
    ap("## OOS lockbox (SSOT §3.1)")
    ap("")
    oos = signed.get("oos", {})
    if isinstance(oos, Mapping):
        ap(f"- **access_count:** {_fmt(oos.get('access_count'))}"
           + ("  ⚠️ CONTAMINATED (> 1)" if _int(oos.get("access_count")) > 1 else ""))
        ap(f"- **oos_source:** {_fmt(oos.get('source'))}")
        ap(f"- **sealed:** {_fmt(oos.get('sealed'))}")
        seal = oos.get("seal")
        if isinstance(seal, Mapping):
            ap(f"- **oos_hash:** `{_fmt(seal.get('oos_hash'))}`")
            ap(f"- **n_oos:** {_fmt(seal.get('n_oos'))}")
        ap(f"- **log_integrity_ok:** {_fmt(oos.get('log_integrity_ok'))}")
        ap("")

    # --- Per-head verdicts ------------------------------------------------- #
    ap("## Per-head verdicts")
    ap("")
    ap("| head | parity | verdict | OOS trades | WF windows | OOS regimes | DSR(worst) | cost-adj CAGR | worst ruin |")
    ap("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for h in signed.get("heads", []) or []:
        if not isinstance(h, Mapping):
            continue
        v = h.get("verdict", {})
        m = h.get("metrics", {})
        v = v if isinstance(v, Mapping) else {}
        m = m if isinstance(m, Mapping) else {}
        ap(
            f"| {_fmt(h.get('head'))} | {_fmt(m.get('parity_status'))} | "
            f"{_fmt(v.get('label') or v.get('verdict'))} | {_fmt(m.get('n_oos_trades'))} | "
            f"{_fmt(m.get('n_walk_forward_windows'))} | {_fmt(m.get('n_regimes_in_oos'))} | "
            f"{_fmt(m.get('deflated_sharpe_worst'))} | {_fmt(m.get('cost_adj_cagr'))} | "
            f"{_fmt(m.get('worst_quantile_ruin'))} |"
        )
    ap("")
    for h in signed.get("heads", []) or []:
        if not isinstance(h, Mapping):
            continue
        v = h.get("verdict", {})
        v = v if isinstance(v, Mapping) else {}
        reasons = v.get("reasons") or []
        ap(f"### {_fmt(h.get('head'))} — {_fmt(v.get('label') or v.get('verdict'))}")
        for r in reasons:
            ap(f"- {r}")
        ap("")

    # --- Portfolio --------------------------------------------------------- #
    portfolio = signed.get("portfolio", {})
    if isinstance(portfolio, Mapping) and portfolio:
        ap("## Portfolio (SSOT §4.2)")
        ap("")
        pv = portfolio.get("verdict", {})
        pv = pv if isinstance(pv, Mapping) else {}
        ap(f"- **verdict:** {_fmt(pv.get('label') or pv.get('verdict'))}")
        ap(f"- **capital in surviving heads:** "
           f"{_fmt(portfolio.get('capital_fraction_in_surviving_heads'))}")
        ap(f"- **surviving heads:** {_fmt(portfolio.get('surviving_heads'))} / "
           f"{_fmt(portfolio.get('total_heads'))}")
        for r in (pv.get("reasons") or []):
            ap(f"- {r}")
        ap("")

    # --- Data quality (Phase 0) ------------------------------------------- #
    ap("## Data quality (Phase 0)")
    ap("")
    dq = signed.get("data_quality", {})
    if isinstance(dq, Mapping):
        ap(f"- **worst_status:** {_fmt(dq.get('worst_status'))}")
        symbols = dq.get("symbols") or []
        if symbols:
            ap("")
            ap("| symbol | status | bars | first | last | quote_ccy |")
            ap("| --- | --- | --- | --- | --- | --- |")
            for s in symbols:
                if not isinstance(s, Mapping):
                    continue
                ap(
                    f"| {_fmt(s.get('symbol'))} | {_fmt(s.get('status'))} | "
                    f"{_fmt(s.get('bar_count'))} | {_fmt(s.get('first_date'))} | "
                    f"{_fmt(s.get('last_date'))} | {_fmt(s.get('quote_currency'))} |"
                )
        ap("")

    # --- Parity map -------------------------------------------------------- #
    ap("## Feature-parity map (SSOT §V1)")
    ap("")
    table = signed.get("parity_table")
    if isinstance(table, str) and table:
        ap("```")
        ap(table)
        ap("```")
        ap("")

    # --- Frozen config ----------------------------------------------------- #
    ap("## Frozen config (SSOT §3.2)")
    ap("")
    fc = signed.get("config_frozen", {})
    if isinstance(fc, Mapping):
        frozen = fc.get("frozen", {})
        frozen = frozen if isinstance(frozen, Mapping) else {}
        ap(f"- **config_hash:** `{_fmt(frozen.get('config_hash'))}`")
        ap(f"- **frozen_at:** {_fmt(frozen.get('frozen_at'))}")
        ap(f"- **trial_count (deflation N add-on):** {_fmt(fc.get('trial_count'))}")
        ap(f"- **last_verdict:** {_fmt(fc.get('last_verdict'))}")
        superseded = fc.get("superseded_hashes") or []
        if superseded:
            ap(f"- **superseded (post-FAIL) hashes:** {len(superseded)}")
        ap("")

    # --- Signature --------------------------------------------------------- #
    sig = signed.get("signature", {})
    if isinstance(sig, Mapping) and sig:
        ap("## Signature")
        ap("")
        ap(f"- **algo:** {_fmt(sig.get('algo'))}")
        ap(f"- **content_sha256:** `{_fmt(sig.get('content_sha256'))}`")
        ap("")

    return "\n".join(lines) + "\n"


def _int(value: Any) -> int:
    """Best-effort int for display guards (a missing/None count reads as 0)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def write_report(
    signed: Mapping[str, Any],
    out_dir: Path | str,
    *,
    basename: Optional[str] = None,
) -> tuple[Path, Path]:
    """Write the signed report to ``out_dir`` as ``<basename>.json`` + ``<basename>.md``.

    ``basename`` defaults to :func:`report_basename` over the report's ``generated_at``.
    Writes ONLY those two files (never ``runtime/``). Deterministic: the JSON is
    ``sort_keys``/``indent=2`` and the Markdown is pure formatting, so identical input
    yields byte-identical files. Returns ``(json_path, md_path)``.
    """
    if basename is None:
        gen = signed.get("generated_at")
        if not isinstance(gen, str) or not gen:
            raise ValueError("signed report needs a 'generated_at' to derive a basename")
        basename = report_basename(gen)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{basename}.json"
    md_path = out / f"{basename}.md"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(signed, fh, sort_keys=True, indent=2)
        fh.write("\n")
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(render_markdown(signed))
    return json_path, md_path
