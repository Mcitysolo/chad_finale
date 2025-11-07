# chad/__init__.py
"""
CHAD — Compounding Hedge-Fund Desk (top-level package marker).

This file ensures the `chad` directory is treated as a proper Python package
in all environments (local, CI, systemd). It also exposes a minimal, stable
package interface for Phase-4 scaffolds and future modules.
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

# Bump when releasing meaningful API surface changes.
__version__ = "0.0.1"
