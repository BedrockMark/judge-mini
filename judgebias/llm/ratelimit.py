from __future__ import annotations

import asyncio
import time
from collections import deque

WINDOW = 60.0


class RateLimiter:
    """At most `rpm` request starts in any sliding 60 s window, which also keeps a fixed-window server quota happy."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self._starts: deque[float] = deque()
        self._lock = asyncio.Lock()  # waiters queue FIFO behind the lock

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            while self._starts and now - self._starts[0] >= WINDOW:
                self._starts.popleft()
            if len(self._starts) >= self.rpm:
                await asyncio.sleep(self._starts[0] + WINDOW - now)
                self._starts.popleft()
            self._starts.append(time.monotonic())
