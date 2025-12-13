#!/usr/bin/env python3
"""
chad/utils/event_bus.py

Lightweight in-process event bus for CHAD.

Phase-3 goal:
- Give the orchestrator and strategies a simple way to emit structured events
  (for logging, metrics, or future async hooks) without pulling in any heavy
  frameworks.
- Keep it synchronous and in-process for now; later phases can extend this to
  Redis, Kafka, or external log sinks if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, DefaultDict, Dict, List, Mapping, MutableMapping, Optional
from collections import defaultdict


@dataclass(frozen=True)
class Event:
    """
    A single CHAD event.

    Examples:
        - type="signal.generated", payload={"strategy": "alpha", "count": 3}
        - type="orchestrator.cycle", payload={"duration_ms": 120}
        - type="error", payload={"where": "alpha", "message": "..."}
    """

    type: str
    payload: Mapping[str, Any]
    created_at: datetime

    @staticmethod
    def now(event_type: str, payload: Mapping[str, Any]) -> "Event":
        return Event(
            type=event_type,
            payload=dict(payload),
            created_at=datetime.now(timezone.utc),
        )


EventHandler = Callable[[Event], None]


class EventBus:
    """
    Minimal in-process event bus.

    - Subscribers register callbacks for specific event types or for '*'
      (catch-all).
    - publish() is synchronous and will call all handlers in the current thread.

    This is intentionally simple and side-effect free for Phase 3; if a handler
    raises, the error is propagated to the caller (so the orchestrator can
    decide what to do).
    """

    def __init__(self) -> None:
        self._handlers: DefaultDict[str, List[EventHandler]] = defaultdict(list)
        self._catch_all: List[EventHandler] = []

    # ------------------------------------------------------------------ #
    # Subscription API
    # ------------------------------------------------------------------ #

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Subscribe a handler to a specific event type.

        Multiple handlers may subscribe to the same type.
        """
        self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """
        Subscribe a handler to receive all events (catch-all).
        """
        self._catch_all.append(handler)

    # ------------------------------------------------------------------ #
    # Publish API
    # ------------------------------------------------------------------ #

    def publish(self, event: Event) -> None:
        """
        Publish an event. Handlers are invoked synchronously.

        If any handler raises, the exception is propagated to the caller.
        """
        # Specific handlers first
        for handler in list(self._handlers.get(event.type, [])):
            handler(event)

        # Then catch-all handlers
        for handler in list(self._catch_all):
            handler(event)

    def emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        """
        Convenience wrapper to build Event.now(...) and publish it.
        """
        self.publish(Event.now(event_type, payload))


# Convenience singleton used by most of CHAD.
_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """
    Return the process-global EventBus singleton.

    This avoids threading a bus instance through every call site
    while still keeping the implementation simple and testable.
    """
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus
