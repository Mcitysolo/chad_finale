"""
Tests for backend.app (CHAD Backend wrapper – Phase 7).

These tests assert the backend's own routes (registered before the API
Gateway mount) behave correctly.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import app


def test_healthz_liveness_probe() -> None:
    """
    GAP-017A: /healthz must be a lightweight liveness probe returning a
    fixed payload identifying the chad-backend service.
    """
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"status": "ok", "service": "chad-backend"}


def test_healthz_registered_before_gateway_mount() -> None:
    """
    /healthz is owned by backend.app and must not be shadowed by the
    api_gateway mount. The gateway's /health remains independently routed.
    """
    client = TestClient(app)
    assert client.get("/healthz").status_code == 200
    assert client.get("/health").status_code == 200
