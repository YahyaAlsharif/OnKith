"""Shared core state and fan-out to every subscribed surface.

The core process owns playback, speech, and capture state. Terminal and
kiosk surfaces subscribe to this bus; none of them owns the state itself,
so a surface may appear or disappear at any time without interrupting
audio.
"""

from __future__ import annotations

import copy
import threading
from typing import Any, Callable, Iterator

Subscriber = Callable[[dict[str, Any]], None]


_MISSING = object()


def _merge(target: dict[str, Any], patch: dict[str, Any]) -> bool:
    """Apply one shallow-per-key merge and report whether anything changed."""

    changed = False
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            if _merge(target[key], value):
                changed = True
            continue
        if target.get(key, _MISSING) != value:
            target[key] = value
            changed = True
    return changed


class EventBus:
    """Hold the authoritative core state and publish deltas to surfaces."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "assistant": {"state": "idle"},
            "playback": {
                "available": False,
                "duration": 0.0,
                "ducked": False,
                "index": 0,
                "muted": False,
                "playing": False,
                "position": 0.0,
                "state": "idle",
                "volume": 82,
            },
            "conversation": {"turns": []},
            # What became of the last thing anybody said: held on this
            # device, sent to the model, or stopped by the masker. It is a
            # record rather than a spinner, so it stands until the next
            # utterance replaces it.
            "gate": {"state": "held"},
            # Whether the player panel was asked for out loud. None means
            # nobody has said either way and the screen follows the results
            # on its own.
            "panel": {"open": None},
            # Whether the microphone is holding an utterance open right now.
            # A surface cannot work this out for itself: the level topic is
            # the output sink, so it moves for music as readily as for a
            # voice, and the assistant only wakes up once an utterance has
            # already finished.
            "listening": {"hearing": False},
            "level": {"peak": 0.0, "rms": 0.0},
            "tracks": {"index": 0, "items": []},
            "status": {"level": "info", "message": ""},
        }
        self._subscribers: dict[int, Subscriber] = {}
        self._next_subscriber_id = 0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    def subscribe(self, subscriber: Subscriber) -> int:
        with self._lock:
            subscriber_id = self._next_subscriber_id
            self._next_subscriber_id += 1
            self._subscribers[subscriber_id] = subscriber
        return subscriber_id

    def unsubscribe(self, subscriber_id: int) -> None:
        with self._lock:
            self._subscribers.pop(subscriber_id, None)

    def _fan_out(self, message: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers.values())
        for subscriber in subscribers:
            try:
                subscriber(message)
            except Exception:
                # One stalled or closed surface must never stop playback or
                # prevent the remaining surfaces from receiving the update.
                continue

    def publish(self, section: str, patch: dict[str, Any]) -> None:
        """Merge one section patch and fan the merged section out."""

        with self._lock:
            target = self._state.setdefault(section, {})
            if not _merge(target, patch):
                return
            payload = {"type": section, **copy.deepcopy(target)}
        self._fan_out(payload)

    def emit(self, message: dict[str, Any]) -> None:
        """Send one message that is not retained in the snapshot."""

        self._fan_out(dict(message))

    def status(self, message: str, level: str = "info") -> None:
        self.publish("status", {"message": message, "level": level})

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - convenience
        return iter(self.snapshot())
