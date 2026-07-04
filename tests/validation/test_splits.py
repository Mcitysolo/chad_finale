"""Known-answer + leakage tests for chad.validation.splits (Phase 2, SSOT §3.7).

The central property under test is *no label leakage*: with a per-head label
horizon ``h``, a bar at index ``i`` has a label spanning ``[i, i+h]``, so every
train bar must satisfy ``i + h < test_start`` — no training label may be computed
from a bar inside the test/validation range. Several tests adversarially attempt
to construct such a leak and assert the generator prevents it.

Fixtures are pure index arithmetic; nothing here reads a file, the network, or any
runtime state.
"""

from __future__ import annotations

import pytest

from chad.validation.splits import (
    Partition,
    WalkForwardWindow,
    generate_walk_forward,
    partition,
)


# --------------------------------------------------------------------------- #
# 1. Partition — chronological boundaries.
# --------------------------------------------------------------------------- #
def test_partition_ranges_no_purge():
    """n=100, default 0.6/0.2/0.2, horizon 0, embargo 0.

    t = floor(100*0.6)=60 ; v = 60+floor(100*0.2)=80.
    train=[0,60), val=[60,80), oos=[80,100); with no gap all indices are kept.
    """
    p = partition(100, label_horizon=0)
    assert isinstance(p, Partition)
    assert p.train_range == (0, 60)
    assert p.val_range == (60, 80)
    assert p.oos_range == (80, 100)
    assert p.train_indices == tuple(range(0, 60))
    assert p.val_indices == tuple(range(60, 80))
    assert p.oos_indices == tuple(range(80, 100))


def test_partition_purge_and_embargo_pull_back_boundaries():
    """horizon 3, embargo 2 → gap 5 trimmed from train tail and val tail.

    train keeps [0, 60-5)=[0,55); val keeps [60, 80-5)=[60,75); oos full [80,100).
    Last train index 54 → 54+3=57 < val_start 60 (no label reaches val).
    Last val index 74 → 74+3=77 < oos_start 80 (no label reaches oos).
    """
    p = partition(100, label_horizon=3, embargo=2)
    assert p.train_indices == tuple(range(0, 55))
    assert p.val_indices == tuple(range(60, 75))
    assert p.oos_indices == tuple(range(80, 100))
    assert max(p.train_indices) + p.label_horizon < p.val_range[0]
    assert max(p.val_indices) + p.label_horizon < p.oos_range[0]


def test_partition_slices_are_disjoint():
    """train / val / oos usable indices never overlap."""
    p = partition(200, label_horizon=4, embargo=3, train_frac=0.5, val_frac=0.25, oos_frac=0.25)
    tr, va, oo = set(p.train_indices), set(p.val_indices), set(p.oos_indices)
    assert tr & va == set()
    assert va & oo == set()
    assert tr & oo == set()


def test_partition_no_train_label_leaks_into_val_or_oos():
    """Adversarial: for a multi-day horizon, no train label window may touch val/oos."""
    p = partition(300, label_horizon=10, embargo=5)
    val_and_oos = set(p.val_indices) | set(p.oos_indices) | set(range(*p.val_range)) | set(
        range(*p.oos_range)
    )
    for i in p.train_indices:
        label_window = set(range(i, i + p.label_horizon + 1))
        assert label_window & val_and_oos == set(), f"train bar {i} label leaks: {label_window}"


def test_partition_to_dict_is_json_shaped():
    p = partition(50, label_horizon=2)
    d = p.to_dict()
    assert d["train_range"] == [0, 30]
    assert isinstance(d["train_indices"], list)
    assert d["label_horizon"] == 2


# --------------------------------------------------------------------------- #
# 2. Partition — degenerate sizing (sentinels, never raises) + invalid config.
# --------------------------------------------------------------------------- #
def test_partition_tiny_series_warns_not_raises():
    """n so small the gap fully purges train/val → empty tuples + warnings, no raise."""
    p = partition(6, label_horizon=3, embargo=2)  # gap 5 >= each slice
    assert p.train_indices == ()   # 0..(3 or so) fully purged
    assert isinstance(p.warnings, tuple) and p.warnings  # at least one warning
    # oos is the tail and is never purged.
    assert p.oos_indices == tuple(range(p.oos_range[0], 6))


def test_partition_invalid_config_raises():
    with pytest.raises(ValueError):
        partition(0, label_horizon=1)                       # n < 1
    with pytest.raises(ValueError):
        partition(100, label_horizon=-1)                    # negative horizon
    with pytest.raises(ValueError):
        partition(100, label_horizon=0, embargo=-2)         # negative embargo
    with pytest.raises(ValueError):
        partition(100, label_horizon=0, train_frac=0.7, val_frac=0.2, oos_frac=0.2)  # sum != 1
    with pytest.raises(ValueError):
        partition(100, label_horizon=0, train_frac=1.0, val_frac=0.0, oos_frac=0.0)  # frac not in (0,1)
    with pytest.raises(ValueError):
        partition(100.0, label_horizon=0)                   # non-int n
    with pytest.raises(ValueError):
        partition(100, label_horizon=True)                  # bool masquerading as int


# --------------------------------------------------------------------------- #
# 3. Walk-forward — known window sequence.
# --------------------------------------------------------------------------- #
def test_walk_forward_known_windows_rolling():
    """n=100, train=40, test=10, horizon=3, embargo=2, step=default(=test=10).

    gap = 5. first_test_start = 40+5 = 45.
    window k: test_start = 45+10k, test_end = 55+10k, train_end = test_start-5,
              rolling train_start = train_end-40.
      k=0: test[45,55) train_end 40 train[0,40)
      k=1: test[55,65) train_end 50 train[10,50)
      k=2: test[65,75) train_end 60 train[20,60)
      k=3: test[75,85) train_end 70 train[30,70)
      k=4: test[85,95) train_end 80 train[40,80)
      k=5: test_end 105 > 100 → stop. → 5 windows.
    """
    ws = generate_walk_forward(100, train_size=40, test_size=10, label_horizon=3, embargo=2)
    assert len(ws) == 5
    assert isinstance(ws[0], WalkForwardWindow)
    assert [(w.test_start, w.test_end) for w in ws] == [
        (45, 55), (55, 65), (65, 75), (75, 85), (85, 95)
    ]
    assert [(w.train_start, w.train_end) for w in ws] == [
        (0, 40), (10, 50), (20, 60), (30, 70), (40, 80)
    ]
    for w in ws:
        assert w.purged_count == 5  # gap = horizon + embargo
        assert w.train_indices == tuple(range(w.train_start, w.train_end))
        assert w.test_indices == tuple(range(w.test_start, w.test_end))


def test_walk_forward_expanding_anchors_train_at_zero():
    """Expanding mode keeps train_start at 0 and grows the train block each step."""
    ws = generate_walk_forward(
        100, train_size=30, test_size=10, label_horizon=2, embargo=0, mode="expanding"
    )
    assert all(w.train_start == 0 for w in ws)
    # train_end grows monotonically with the window index.
    ends = [w.train_end for w in ws]
    assert ends == sorted(ends) and len(set(ends)) == len(ends)


def test_walk_forward_custom_step_controls_advance():
    """A custom step overrides the default (=test_size) advance between windows."""
    ws = generate_walk_forward(
        100, train_size=20, test_size=10, label_horizon=1, embargo=0, step=5
    )
    starts = [w.test_start for w in ws]
    # first_test_start = 20+1 = 21 ; step 5 → 21,26,31,...
    assert starts[0] == 21
    assert starts[1] - starts[0] == 5


# --------------------------------------------------------------------------- #
# 4. Walk-forward — the leakage guarantee (adversarial).
# --------------------------------------------------------------------------- #
def test_walk_forward_train_and_test_are_disjoint():
    ws = generate_walk_forward(120, train_size=30, test_size=15, label_horizon=4, embargo=3)
    assert ws
    for w in ws:
        assert set(w.train_indices) & set(w.test_indices) == set()


def test_walk_forward_no_train_label_reaches_test():
    """Adversarial construction: for every train bar i, its label window [i, i+h]
    must NOT intersect the test range. This is the leakage the purge exists to stop.
    """
    for h in (1, 3, 7, 12):
        ws = generate_walk_forward(
            200, train_size=50, test_size=20, label_horizon=h, embargo=2
        )
        assert ws, f"expected windows for horizon {h}"
        for w in ws:
            test_set = set(w.test_indices)
            for i in w.train_indices:
                label_window = set(range(i, i + h + 1))
                assert label_window & test_set == set(), (
                    f"leak at horizon {h}, window {w.index}: train bar {i} "
                    f"label {sorted(label_window)} touches test {w.test_start}..{w.test_end}"
                )
            # Tight boundary form of the same property.
            assert max(w.train_indices) + h < w.test_start


def test_walk_forward_bigger_horizon_purges_more():
    """Increasing the label horizon widens the train→test gap by exactly that much."""
    small = generate_walk_forward(200, train_size=50, test_size=20, label_horizon=2, embargo=1)
    big = generate_walk_forward(200, train_size=50, test_size=20, label_horizon=8, embargo=1)
    # Same test_start for window 0 shifts by the horizon difference (2→8 = +6).
    assert big[0].test_start - small[0].test_start == 6
    assert small[0].purged_count == 3 and big[0].purged_count == 9


# --------------------------------------------------------------------------- #
# 5. Walk-forward — degenerate sizing (sentinel) + invalid config + determinism.
# --------------------------------------------------------------------------- #
def test_walk_forward_too_short_returns_empty():
    """train+gap+test > n → no window fits → [] (documented sentinel, no raise)."""
    ws = generate_walk_forward(30, train_size=25, test_size=10, label_horizon=2, embargo=1)
    assert ws == []


def test_walk_forward_invalid_config_raises():
    with pytest.raises(ValueError):
        generate_walk_forward(100, train_size=0, test_size=10, label_horizon=1)
    with pytest.raises(ValueError):
        generate_walk_forward(100, train_size=10, test_size=0, label_horizon=1)
    with pytest.raises(ValueError):
        generate_walk_forward(100, train_size=10, test_size=10, label_horizon=-1)
    with pytest.raises(ValueError):
        generate_walk_forward(100, train_size=10, test_size=10, label_horizon=1, embargo=-1)
    with pytest.raises(ValueError):
        generate_walk_forward(100, train_size=10, test_size=10, label_horizon=1, step=0)
    with pytest.raises(ValueError):
        generate_walk_forward(100, train_size=10, test_size=10, label_horizon=1, mode="sideways")


def test_walk_forward_determinism():
    """Same inputs → identical window sequence (dict-equal)."""
    a = generate_walk_forward(150, train_size=40, test_size=15, label_horizon=5, embargo=2)
    b = generate_walk_forward(150, train_size=40, test_size=15, label_horizon=5, embargo=2)
    assert [w.to_dict() for w in a] == [w.to_dict() for w in b]
