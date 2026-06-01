"""Unit tests for the reversible futures-execution disable env gate.

Covers chad.core.live_loop._futures_execution_disabled across the three
recognised flags, their truthy/falsy vocabularies, and edge cases
(unset, empty, whitespace, mixed case).
"""

import pytest

from chad.core.live_loop import _futures_execution_disabled


def test_all_flags_unset_is_false():
    assert _futures_execution_disabled({}) is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "YES", "on", "On"])
def test_disable_futures_execution_truthy(value):
    assert _futures_execution_disabled(
        {"CHAD_DISABLE_FUTURES_EXECUTION": value}
    ) is True


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_disable_futures_truthy(value):
    assert _futures_execution_disabled({"CHAD_DISABLE_FUTURES": value}) is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", "Off"])
def test_futures_execution_enabled_falsy_disables(value):
    assert _futures_execution_disabled(
        {"CHAD_FUTURES_EXECUTION_ENABLED": value}
    ) is True


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_futures_execution_enabled_truthy_does_not_disable(value):
    assert _futures_execution_disabled(
        {"CHAD_FUTURES_EXECUTION_ENABLED": value}
    ) is False


@pytest.mark.parametrize(
    "env",
    [
        {"CHAD_DISABLE_FUTURES_EXECUTION": ""},
        {"CHAD_DISABLE_FUTURES": ""},
        {"CHAD_FUTURES_EXECUTION_ENABLED": ""},
    ],
)
def test_empty_strings_are_false(env):
    assert _futures_execution_disabled(env) is False


def test_whitespace_and_case_disable_truthy():
    assert _futures_execution_disabled(
        {"CHAD_DISABLE_FUTURES_EXECUTION": "  Yes  "}
    ) is True


def test_whitespace_and_case_enabled_falsy():
    assert _futures_execution_disabled(
        {"CHAD_FUTURES_EXECUTION_ENABLED": "  OFF  "}
    ) is True


def test_unrecognised_value_is_false():
    # A non-vocabulary value on the enable flag must not disable (fail-open
    # only for garbage; explicit falsy/truthy handled elsewhere).
    assert _futures_execution_disabled(
        {"CHAD_FUTURES_EXECUTION_ENABLED": "maybe"}
    ) is False


def test_disable_flag_wins_when_enable_also_present():
    assert _futures_execution_disabled(
        {
            "CHAD_DISABLE_FUTURES": "1",
            "CHAD_FUTURES_EXECUTION_ENABLED": "1",
        }
    ) is True
