"""Unit tests for scripts/dismiss_paper_account_notice.py.

Hermetic: exercises ONLY the pure parse/geometry/artifact helpers. No xdotool
process is spawned, no X server is touched. The module is loaded directly from
its file path so the tests do not depend on ``scripts`` being an importable
package.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "dismiss_paper_account_notice.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("dismiss_paper_account_notice", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dpan = _load_module()


# --- parse_window_geometry ---------------------------------------------------

# Exact stdout captured live from `xdotool getwindowgeometry 12583125` on :99.
LIVE_GEOMETRY_OUTPUT = "Window 12583125\n  Position: 638,453 (screen: 0)\n  Geometry: 644x175\n"


def test_parse_geometry_live_window():
    geom = dpan.parse_window_geometry(LIVE_GEOMETRY_OUTPUT)
    assert geom == dpan.Geometry(x=638, y=453, w=644, h=175)


def test_parse_geometry_without_screen_suffix():
    out = "Window 1\n  Position: 100,200\n  Geometry: 300x150\n"
    assert dpan.parse_window_geometry(out) == dpan.Geometry(x=100, y=200, w=300, h=150)


def test_parse_geometry_negative_position():
    out = "Window 1\n  Position: -5,-12 (screen: 0)\n  Geometry: 644x175\n"
    assert dpan.parse_window_geometry(out) == dpan.Geometry(x=-5, y=-12, w=644, h=175)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "Window 1\n  Geometry: 644x175\n",          # missing Position
        "Window 1\n  Position: 638,453 (screen: 0)\n",  # missing Geometry
        "garbage",
    ],
)
def test_parse_geometry_rejects_malformed(bad):
    with pytest.raises(ValueError):
        dpan.parse_window_geometry(bad)


# --- compute_click_target ----------------------------------------------------


def test_click_target_for_live_window_is_960_606():
    geom = dpan.Geometry(x=638, y=453, w=644, h=175)
    assert dpan.compute_click_target(geom) == (960, 606)


def test_click_target_formula():
    geom = dpan.Geometry(x=10, y=20, w=100, h=60)
    # (10 + 100//2, 20 + 60 - 22) = (60, 58)
    assert dpan.compute_click_target(geom) == (60, 58)


def test_click_target_uses_floor_division_on_odd_width():
    geom = dpan.Geometry(x=0, y=0, w=645, h=175)
    # 645 // 2 == 322 (not 322.5)
    assert dpan.compute_click_target(geom) == (322, 153)


def test_button_offset_constant():
    assert dpan.BUTTON_OFFSET_FROM_BOTTOM_PX == 22


# --- parse_search_output -----------------------------------------------------


def test_parse_search_single_wid():
    assert dpan.parse_search_output("12583125\n") == [12583125]


def test_parse_search_multiple_and_blank_lines():
    assert dpan.parse_search_output("111\n\n222\n  333  \n") == [111, 222, 333]


def test_parse_search_ignores_non_numeric_noise():
    assert dpan.parse_search_output("xdotool: no match\n12583125\n") == [12583125]


def test_parse_search_empty():
    assert dpan.parse_search_output("") == []


# --- build_artifact ----------------------------------------------------------


def test_build_artifact_dry_run_shape():
    art = dpan.build_artifact(
        ts_utc="2026-06-24T20:46:00Z",
        display=":99",
        notice_seen=True,
        dismissed=False,
        attempts=0,
        status="DRY_RUN",
        geometry=dpan.Geometry(x=638, y=453, w=644, h=175),
        click_target=(960, 606),
    )
    assert art == {
        "ts_utc": "2026-06-24T20:46:00Z",
        "display": ":99",
        "notice_seen": True,
        "dismissed": False,
        "attempts": 0,
        "status": "DRY_RUN",
        "geometry": {"x": 638, "y": 453, "w": 644, "h": 175},
        "click_target": [960, 606],
    }


def test_build_artifact_not_seen_nulls():
    art = dpan.build_artifact(
        ts_utc="2026-06-24T20:46:00Z",
        display=":99",
        notice_seen=False,
        dismissed=False,
        attempts=0,
        status="NOT_SEEN",
        geometry=None,
        click_target=None,
    )
    assert art["geometry"] is None
    assert art["click_target"] is None
    assert art["status"] == "NOT_SEEN"


def test_artifact_keys_are_stable_contract():
    art = dpan.build_artifact(
        ts_utc="t",
        display=":99",
        notice_seen=False,
        dismissed=False,
        attempts=0,
        status="NOT_SEEN",
        geometry=None,
        click_target=None,
    )
    assert set(art.keys()) == {
        "ts_utc",
        "display",
        "notice_seen",
        "dismissed",
        "attempts",
        "status",
        "geometry",
        "click_target",
    }


# --- module invariants (safety contract) -------------------------------------


def test_only_target_title_literal():
    assert dpan.TARGET_TITLE == "Paper Account Notice"


def test_bounds_are_finite_and_sane():
    assert dpan.POLL_BUDGET_SECONDS == 120.0
    assert dpan.POLL_INTERVAL_SECONDS == 3.0
    assert dpan.MAX_CLICK_ATTEMPTS == 3
    assert dpan.SUBPROC_TIMEOUT_SECONDS > 0


def test_artifact_dir_is_under_repo_reports():
    assert dpan.ARTIFACT_DIR.as_posix().endswith("reports/gateway_paper_notice_log")
