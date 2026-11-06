from __future__ import annotations

import asyncio
import heapq
import itertools
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


class PriorityGate:
    """Semaphore that hands freed slots to the waiter with the lowest priority value (FIFO among equals)."""

    def __init__(self, slots: int):
        self._free = slots
        self._waiters: list[tuple[int, int, asyncio.Future]] = []
        self._seq = itertools.count()

    async def acquire(self, priority: int = 0) -> None:
        if self._free > 0 and not self._waiters:
            self._free -= 1
            return
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (priority, next(self._seq), fut))
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():  # the slot was handed over just before cancellation
                self.release()
            raise

    def release(self) -> None:
        while self._waiters:
            fut = heapq.heappop(self._waiters)[2]
            if not fut.done():  # skip waiters cancelled while queued
                fut.set_result(None)
                return
        self._free += 1
