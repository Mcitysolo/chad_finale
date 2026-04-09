"""
chad/utils/runtime_defaults.py

Centralized helpers for canonical runtime file paths used across CHAD.
Currently provides only the dynamic caps path; additional defaults can
be added here as needed.
"""

from __future__ import annotations

from pathlib import Path


def chad_repo_root() -> Path:
    """Return the canonical CHAD repo root."""
    return Path(__file__).resolve().parents[2]


def default_dynamic_caps_path() -> Path:
    """Canonical path to runtime/dynamic_caps.json."""
    return chad_repo_root() / "runtime" / "dynamic_caps.json"
