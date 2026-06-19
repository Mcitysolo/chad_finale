"""
L-05 regression test: backend.ai_surface._client() must return ONE
process-lifetime GPTClient so a single requests.Session
(chad/intel/gpt_client.py:238) is reused across requests.

Before the fix, `_client()` built a fresh GPTClient() per call (ai_surface.py
~:104), each with its own requests.Session that is never closed (no close(),
__del__, or __enter__/__exit__ on GPTClient; the Session is only used at
gpt_client.py:496). Every advisory request therefore leaked a keep-alive socket
on the long-lived backend process.

This module proves two things and REPORTS both arms:
  1. Identity   -- cached `_client()` returns the SAME instance N times,
                   sharing ONE Session; the *uncached* function
                   (`_client.__wrapped__`) returns DISTINCT instances with
                   DISTINCT Sessions (the original leak).
  2. Socket fds -- exercising the cached client's Session N times against a
                   local keep-alive server keeps the open-socket count FLAT,
                   whereas the uncached (fresh-client-per-call) pattern grows
                   the open-socket count ~linearly with N.

The fd arm is what makes this catch a regression: if the lru_cache were
removed, `_client()` would behave like the uncached arm and the identity
assertion below would fail.

No network, no runtime/ writes -- a throwaway localhost HTTP server stands in
for the OpenAI endpoint (gpt_client.OPENAI_CHAT_URL is module-level; we drive
the client's real `_session` directly, which is the exact resource L-05 leaks).
"""

from __future__ import annotations

import http.server
import os
import socketserver
import threading

import pytest

import backend.ai_surface as ai_surface


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _count_socket_fds() -> int:
    """Count open socket file descriptors for THIS process (Linux /proc)."""
    count = 0
    fd_dir = "/proc/self/fd"
    for name in os.listdir(fd_dir):
        try:
            target = os.readlink(os.path.join(fd_dir, name))
        except OSError:
            continue  # fd vanished between listdir and readlink
        if target.startswith("socket:"):
            count += 1
    return count


class _KeepAliveHandler(http.server.BaseHTTPRequestHandler):
    # HTTP/1.1 => persistent (keep-alive) connections, so a reused Session
    # keeps a single pooled socket open and a fresh Session opens a new one.
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 (stdlib signature)
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs) -> None:  # silence test noise
        return


@pytest.fixture()
def local_keepalive_server():
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _KeepAliveHandler)
    server.daemon_threads = True
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def _dummy_openai_key(monkeypatch):
    # GPTClient construction only requires OPENAI_API_KEY to be present; it does
    # not contact OpenAI until a request is made (and we never make one here).
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-l05")
    # Start every test from a clean cache and leave a clean cache behind so the
    # dummy-keyed singleton never bleeds into the rest of the suite.
    ai_surface._client.cache_clear()
    yield
    ai_surface._client.cache_clear()


# --------------------------------------------------------------------------- #
# 1) Identity: singleton vs the original per-call leak
# --------------------------------------------------------------------------- #
def test_client_is_process_singleton() -> None:
    N = 6

    # Cached form (the fix): same instance + same Session every call.
    cached = [ai_surface._client() for _ in range(N)]
    first = cached[0]
    assert all(c is first for c in cached), "cached _client() must return ONE instance"
    assert all(c._session is first._session for c in cached), "must reuse ONE Session"

    # Uncached form (the pre-fix behaviour): distinct instances + distinct
    # Sessions -- i.e. the leak this test is designed to catch.
    raw = ai_surface._client.__wrapped__
    a, b = raw(), raw()
    assert a is not b, "uncached path builds a fresh client per call (the leak)"
    assert a._session is not b._session, "uncached path builds a fresh Session per call"

    # Clean up the Sessions created by the uncached arm.
    a._session.close()
    b._session.close()

    print(
        f"\n[L-05 identity] cached: {len({id(c) for c in cached})} unique instance "
        f"across {N} calls (expect 1) | uncached: {len({id(a), id(b)})} unique "
        f"across 2 calls (expect 2 -> the leak)"
    )


# --------------------------------------------------------------------------- #
# 2) Socket fds: flat under the singleton, growing under the per-call leak
# --------------------------------------------------------------------------- #
def test_ai_surface_calls_do_not_leak_sockets(local_keepalive_server) -> None:
    url = local_keepalive_server
    N = 6

    # --- Leak arm: fresh client per "request", Sessions never closed --------- #
    leaked = []  # hold refs so the leaked Sessions/sockets stay open & countable
    base_leak = _count_socket_fds()
    raw = ai_surface._client.__wrapped__
    for _ in range(N):
        c = raw()                       # fresh GPTClient -> fresh Session
        leaked.append(c)
        c._session.get(url, timeout=5).content  # keep-alive socket, never closed
    leak_growth = _count_socket_fds() - base_leak

    # --- Fixed arm: one cached client reused across N "requests" ------------- #
    ai_surface._client.cache_clear()
    c0 = ai_surface._client()
    base_fixed = _count_socket_fds()
    for _ in range(N):
        c = ai_surface._client()        # same instance every call (the endpoints' pattern)
        assert c is c0
        c._session.get(url, timeout=5).content  # one pooled keep-alive socket, reused
    fixed_growth = _count_socket_fds() - base_fixed

    print(
        f"\n[L-05 sockets] N={N} | leak arm (fresh client/call) grew open sockets "
        f"by {leak_growth} | fixed arm (cached singleton) grew by {fixed_growth}"
    )

    # Clean up leaked Sessions so the rest of the suite sees flat fds.
    for c in leaked:
        c._session.close()
    c0._session.close()

    # The per-call leak grows sockets ~linearly with N (>= one leaked socket per
    # call; in-process server side roughly doubles it). The singleton stays in a
    # small constant band regardless of N.
    assert leak_growth >= N, (
        f"methodology check: per-call leak should grow sockets by >= {N}, "
        f"saw {leak_growth}"
    )
    assert fixed_growth <= 4, (
        f"singleton must keep open sockets flat, grew by {fixed_growth}"
    )
    assert leak_growth > fixed_growth, "leak arm must out-grow the fixed arm"
