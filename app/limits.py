"""Limits on questions that use the LLM, so one visitor (or a script) can't use up the free quota.

Two windows, checked together: per client (its IP address) and for everyone at once. They live in memory, so they
reset when the server restarts; that is enough for a demo server with one process. The Streamlit demo, which has
no client address, uses the database count instead (db.count_questions), which also covers every visitor.
"""
import os
import threading
import time
from collections import deque

PER_CLIENT_PER_MINUTE = int(os.getenv("MAX_QUESTIONS_PER_CLIENT_PER_MINUTE", "6"))
PER_HOUR = int(os.getenv("MAX_QUESTIONS_PER_HOUR", "60"))      # everyone together
PER_DAY = int(os.getenv("MAX_QUESTIONS_PER_DAY", "300"))       # everyone together


class Window:
    """At most `limit` events in the last `seconds`, per key."""

    def __init__(self, limit: int, seconds: float):
        self.limit, self.seconds = limit, seconds
        self._events: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str = "", now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and events[0] <= now - self.seconds:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(now)
            if len(self._events) > 10_000:        # forget idle clients, so the table can't grow without end
                for k in [k for k, v in self._events.items() if not v]:
                    del self._events[k]
            return True


_client = Window(PER_CLIENT_PER_MINUTE, 60)
_hour = Window(PER_HOUR, 3600)
_day = Window(PER_DAY, 86400)


def check(client: str) -> str | None:
    """None if this client may ask now, else the reason it may not (the question is not counted then)."""
    if not _client.allow(client):
        return "Too many questions from you in the last minute. Wait a moment and try again."
    if not _hour.allow():
        return "The demo has had many questions in the last hour. Try again later."
    if not _day.allow():
        return "The demo has reached its daily limit of questions. Try again tomorrow."
    return None
