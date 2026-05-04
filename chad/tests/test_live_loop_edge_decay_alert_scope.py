"""Regression test for the live_loop._EDGE_DECAY_ALERTED scoping bug.

Reproduces the failure mode observed in paper logs:
    cannot access local variable '_EDGE_DECAY_ALERTED' where it is not
    associated with a value

The bug was that ``_EDGE_DECAY_ALERTED -= ...`` inside ``run_once``
made the compiler treat the module-global as a function local for the
entire frame, so the earlier ``in`` / ``.add`` reads raised
``UnboundLocalError``. The fix replaces the augmented assignment with
``.intersection_update(...)`` (in-place mutation, no rebind).
"""

from __future__ import annotations

import ast
from pathlib import Path

import chad.core.live_loop as live_loop


LIVE_LOOP_SRC = Path(live_loop.__file__).read_text(encoding="utf-8")


def test_edge_decay_alerted_is_module_global_not_locally_rebound() -> None:
    """The cleanup path must not use augmented assignment on the global."""
    # The module-level definition exists.
    assert hasattr(live_loop, "_EDGE_DECAY_ALERTED")
    assert isinstance(live_loop._EDGE_DECAY_ALERTED, set)

    # No augmented assignment anywhere on _EDGE_DECAY_ALERTED — that is
    # the construct that promoted it to a frame-local and triggered the
    # UnboundLocalError.
    tree = ast.parse(LIVE_LOOP_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            assert node.target.id != "_EDGE_DECAY_ALERTED", (
                "augmented assignment on _EDGE_DECAY_ALERTED reintroduces "
                "the run_once UnboundLocalError; use a mutation method "
                "such as .intersection_update() instead."
            )


def test_edge_decay_alerted_cleanup_path_does_not_raise() -> None:
    """Exercise the exact mutation pattern run_once uses on the global.

    If anyone reverts to ``-=`` (or similar) and the global gets touched
    inside a function that also reads it earlier, this would explode.
    Here we pin the in-place semantics: read, .add, then
    .intersection_update — no rebinding, no UnboundLocalError.
    """
    # Snapshot and restore so the test is hermetic.
    saved = set(live_loop._EDGE_DECAY_ALERTED)
    try:
        live_loop._EDGE_DECAY_ALERTED.clear()
        live_loop._EDGE_DECAY_ALERTED.add("delta")
        live_loop._EDGE_DECAY_ALERTED.add("test_strategy_bg10_proof")

        # Operator clears delta only — the cleanup line should retain
        # only the still-halted entry.
        currently_halted = {"test_strategy_bg10_proof"}
        live_loop._EDGE_DECAY_ALERTED.intersection_update(currently_halted)

        assert "delta" not in live_loop._EDGE_DECAY_ALERTED
        assert "test_strategy_bg10_proof" in live_loop._EDGE_DECAY_ALERTED
    finally:
        live_loop._EDGE_DECAY_ALERTED.clear()
        live_loop._EDGE_DECAY_ALERTED.update(saved)


def test_run_once_frame_uses_global_consistently() -> None:
    """All references to _EDGE_DECAY_ALERTED inside run_once must be
    method/attribute access or membership tests — never a Store/AugStore
    that would shadow the module global."""
    tree = ast.parse(LIVE_LOOP_SRC)
    run_once = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "run_once"),
        None,
    )
    assert run_once is not None, "run_once not found in live_loop"

    for node in ast.walk(run_once):
        # Disallow plain assignment too (e.g. ``_EDGE_DECAY_ALERTED = ...``).
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "_EDGE_DECAY_ALERTED":
                    raise AssertionError(
                        "run_once assigns to _EDGE_DECAY_ALERTED; this "
                        "shadows the module global and will cause "
                        "UnboundLocalError on prior reads."
                    )
        if isinstance(node, ast.AugAssign):
            tgt = node.target
            if isinstance(tgt, ast.Name) and tgt.id == "_EDGE_DECAY_ALERTED":
                raise AssertionError(
                    "run_once augments _EDGE_DECAY_ALERTED; use "
                    ".intersection_update() / .difference_update() "
                    "instead to keep it a module global."
                )
