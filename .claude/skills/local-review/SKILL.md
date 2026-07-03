---
name: local-review
description: CHAD-specific local multi-agent code review — the free, unlimited local equivalent of the paid cloud /ultrareview (which can't run here because this 15G repo exceeds the teleport size limit). Dispatches four READ-ONLY subagents in parallel (Security, Correctness, Design/Architecture, Verifier) over a scope you name, or over `git diff HEAD` if no scope is given, and synthesizes one report. Encodes CHAD's real invariants: paper-only broker safety, currency provenance, money-truth single-source, Bug-A broker-executor reuse, verified-facts-only. Invoke with /local-review (whole diff) or /local-review <file-or-module>. Never mutates anything.
---

# CHAD Local Review

A local, free, unlimited stand-in for the paid cloud `/ultrareview`. The cloud reviewer
teleports the repo to a build host; this repo is ~15G and exceeds that size limit, so the
cloud path is unavailable. This skill reproduces the same multi-agent adversarial review
**entirely on this box** — no upload, no size limit, no cost.

It is **strictly READ-ONLY**. It reads code and reports. It never edits files, never stages,
never commits, never restarts a service, never touches runtime state. It produces a report;
acting on that report is a separate, human-authorized step.

---

## When invoked

1. **Resolve the scope.**
   - `/local-review <scope>` → review that file, directory, or module (e.g. `chad/risk/profit_lock.py`
     or `chad/execution/`).
   - `/local-review` with no scope → review the working diff: `git diff HEAD` (fall back to
     `git diff --staged` then the last commit `git show HEAD` only if `git diff HEAD` is empty).
   - Before dispatching, run the appropriate read-only command once yourself (`git diff HEAD`,
     or `git ls-files <scope>` + `wc -l`) so you can hand each subagent the exact file list /
     diff text and they don't each re-derive it.

2. **Dispatch FOUR subagents IN PARALLEL** — issue all four `Task` tool calls **in a single
   message** so they run concurrently. Use the `Explore` agent type (read-only) or
   `general-purpose`; whichever you choose, the subagent prompt MUST open with:

   > You are a READ-ONLY reviewer. You may Read, Grep, Glob, and run read-only Bash
   > (`git diff`, `git log`, `grep`, `rg`, `sed -n`). You must NOT edit, write, stage,
   > commit, restart any service, or mutate any runtime/ file. Report findings only, each
   > anchored to a concrete `file:line`. If you cannot cite a `file:line`, do not report it.

   Give each subagent (a) the resolved scope / diff, and (b) its checklist below.

3. **Synthesize.** When all four return, the Verifier's confirmed set is authoritative.
   Emit exactly one report in the OUTPUT FORMAT below. Do not act on any finding.

---

## Subagent 1 — 🔴 Security Reviewer

Review the scope for anything that weakens CHAD's paper-only, jurisdiction-safe posture.
Flag, with `file:line`:

- **Plaintext secrets.** Any IBKR / Kraken / broker credential, API key, session token, or
  account number stored, logged, printed, or committed in plaintext (including into a log
  line, an f-string, an exception message, or a runtime JSON file).
- **Any path that could reach a LIVE order.** Specifically: code that writes
  `ready_for_live = true`, flips `exec_mode`/`CHAD_EXECUTION_MODE` off `paper`, sets
  `ibkr_dry_run = false`, sets `paper_only = false`, or otherwise removes a paper/dry-run
  guard on an execution path.
- **Bypassing the order guard.** Any change that evades `.claude/hooks/chad-order-guard.sh`
  (which blocks `flatten_futures_oneshot`, `micro_eod_flatten`, `kraken_roundtrip_runner`,
  `reqGlobalCancel`) — e.g. renaming/wrapping a blocked entrypoint, invoking it via a
  different shell path, or adding a new live-order/flatten entrypoint the guard doesn't cover.
- **Weakening the futures gate.** Any removal, inversion, default-flip, or narrowing of the
  `CHAD_DISABLE_FUTURES_EXECUTION` env gate. This is permanent defense-in-depth — Kraken
  Futures is jurisdiction-blocked. It must stay fail-closed.
- **Unguarded irreversible broker actions.** Any new outbound or irreversible broker call
  (order submit, cancel, transfer, flatten) that lacks a paper-mode guard or an authorization
  gate ahead of it.
- **Credential drive-by.** Any change that touches auth, credentials, key handling, or
  connection config as a side-effect of an otherwise unrelated task — flag the scope creep.

## Subagent 2 — 🟠 Correctness Reviewer

Review the scope for correctness defects. CHAD's #1 recurring bug class is currency
provenance — weight it first. Flag, with `file:line`:

- **Currency provenance.** Any equity/PnL value that mixes or mislabels CAD vs USD; any
  `*_usd`-named field that actually holds a CAD value (or vice-versa); any hardcoded FX guess
  (e.g. `0.73`, `1.35`, an inline `1.416`); any USD↔CAD conversion that does **not** route
  through the canonical `chad.constants.fx.USDCAD_CONVERSION_CONSTANT` (= 1.4160). A conversion
  factor typed as a literal anywhere but `chad/constants/fx.py` is a finding.
- **Bug-A broker-executor regression.** Any broker-call pattern that mints a per-call thread,
  event loop, or `asyncio.new_event_loop()` / `run_until_complete` instead of using the shared
  `chad.execution.broker_executor` (bounded shared pool, persistent per-worker loops). Per-call
  loop/thread creation is the Bug-A fd/loop-leak regression.
- **Idempotency gaps.** Any submit / dedup / order / fill path where re-running the same input
  would double-submit, double-count, or double-close. Ask concretely: "if this runs twice with
  the same `idempotency_key`/`execution_id`, does it stay single-effect?"
- **Swallowed exceptions.** Any bare `except:` or `except Exception: pass` (or log-and-continue
  on a money/risk path) that hides a failure the caller needed to see.
- **Silent truth override.** Any code that silently overrides SCR (where
  `scr_state.stats.effective_trades` is authoritative) or broker-authority truth
  (`positions_truth` / `positions_snapshot`) instead of **flagging** the discrepancy. Truth
  sources reconcile loudly; they don't get quietly overwritten.
- **Fail-OPEN where fail-CLOSED is required.** Any money/risk/execution path that proceeds
  (permits a trade, sizes up, marks ready) when required state is missing, stale, or
  unavailable. On money paths, missing/stale ⇒ must fail closed.

## Subagent 3 — 🔵 Design / Architecture Reviewer

Review the scope for structural and governance defects. Flag, with `file:line`:

- **Money-truth single source.** Any new writer of an equity/PnL value that creates a
  multi-writer race on a canonical key (the dynamic_caps / tier / snapshot lesson — one owner
  per canonical money key, everyone else reads).
- **Uncanonical constants (Principle 9).** Any magic number or constant re-declared inline
  when a canonical module owns it — e.g. `1.4160` must be imported from `chad/constants/fx.py`,
  not retyped. Same for band edges, tier thresholds, TTLs with a canonical home.
- **Unsafe staging.** Any `git add -A`, `git add .`, `git commit -a`, or other non-explicit-path
  staging in a script — staging must name explicit paths.
- **Rebuild instead of extend.** Any change that rewrites/replaces a working component when it
  should be extended additively. CHAD's rule is "wire/extend, don't rebuild" — flag full
  rewrites of working code.
- **Decisions on unverified claims (verified-facts-only).** Any logic, gate, or comment that
  acts on a document/handoff/SSOT claim without a `file:line` or box-verified probe behind it.
  A claim in a doc is a hypothesis, not ground truth.
- **Observe-but-doesn't-bind mislabeled as binding.** Any risk control described (in code,
  comment, or config) as "binding" / "enforced" when it only observes or logs and has no
  journal/test evidence that it actually blocks. Flag the overclaim.

## Subagent 4 — Verifier

You receive the raw findings from Reviewers 1–3. Your job is adversarial confirmation:

- **Reproduce EVERY flagged issue against the actual code.** Open the cited `file:line`
  yourself. Confirm the code truly does what the reviewer claims — do not trust a claim you
  have not reproduced.
- **Discard** style preferences, nitpicks, duplicates, and anything you cannot reproduce at a
  concrete `file:line`. A finding that doesn't reproduce is not a finding.
- **Keep** only real, reproduced issues. For each kept issue, confirm the `file:line`, the
  mechanism, and why it matters.
- **Record what you discarded and why.** This is a required guardrail: a big audit is a
  hypothesis, not an authorization. Every dropped claim gets one line explaining the drop
  (unreproducible / style-only / already-guarded-at `file:line` / duplicate).

The Verifier's confirmed set is what appears in the final report.

---

## OUTPUT FORMAT

End the review with exactly this structure (omit a severity section only if it has zero
verified issues; never invent issues to fill a section):

```
## Local Review — [scope]

### 🔴 Security
- [issue] — [file:line] — [why it matters]

### 🟠 Correctness
- [issue] — [file:line] — [why it matters]

### 🔵 Design / Architecture
- [issue] — [file:line] — [why it matters]

### Verified: N issues | Discarded as non-issues: M
```

Under the `Verified: N | Discarded: M` line, briefly list what was discarded and why (one
line each), so the reader can see what the Verifier rejected and on what grounds.

---

## Usage

- `/local-review` — review the whole working diff (`git diff HEAD`).
- `/local-review <scope>` — review a specific file, directory, or module,
  e.g. `/local-review chad/risk/profit_lock.py` or `/local-review chad/execution/`.

This skill is **READ-ONLY**. It dispatches read-only subagents, reads code, and prints a
report. It never edits, stages, commits, restarts a service, or mutates any runtime state.
Acting on any finding is a separate, human-authorized step.
