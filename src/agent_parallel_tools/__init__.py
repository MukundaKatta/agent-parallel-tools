"""
agent-parallel-tools: Run multiple tool calls in parallel, results in original order.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    index: int = 0


@dataclass
class ToolResult:
    call: ToolCall
    result: Any = None
    error: Optional[BaseException] = None
    duration_ms: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def name(self) -> str:
        return self.call.name


class ParallelToolRunner:
    """
    Execute a batch of tool calls concurrently and return results in the
    original call order.

    Usage::

        runner = ParallelToolRunner(max_workers=8)
        runner.register("search", search_fn)
        runner.register("fetch", fetch_fn)

        calls = [
            ToolCall("search", {"query": "python"}),
            ToolCall("fetch", {"url": "https://example.com"}),
        ]
        results = runner.run(calls)
        # results[0].name == "search", results[1].name == "fetch"
    """

    def __init__(self, max_workers: int = 8) -> None:
        self._max_workers = max_workers
        self._registry: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> "ParallelToolRunner":
        self._registry[name] = fn
        return self

    def run(
        self, calls: list[ToolCall], timeout: Optional[float] = None
    ) -> list[ToolResult]:
        """
        Execute all calls concurrently. Returns one :class:`ToolResult` per call,
        in the original call order.

        ``timeout`` (seconds) is an overall deadline for the whole batch. Calls
        that have not finished by the deadline are reported with a
        :class:`TimeoutError` instead of being dropped, so the returned list
        always has the same length and order as ``calls``.

        Raises ``KeyError`` for unregistered tools (before starting any threads).
        """
        for call in calls:
            if call.name not in self._registry:
                raise KeyError(f"Tool not registered: {call.name!r}")

        if not calls:
            return []

        results: list[Optional[ToolResult]] = [None] * len(calls)
        workers = min(self._max_workers, len(calls))
        work_q: queue.Queue[ToolCall] = queue.Queue()
        for i, call in enumerate(calls):
            call.index = i
            work_q.put(call)

        def worker() -> None:
            while True:
                try:
                    call = work_q.get_nowait()
                except queue.Empty:
                    break
                fn = self._registry[call.name]
                t0 = time.monotonic()
                try:
                    result = fn(**call.args)
                    duration = (time.monotonic() - t0) * 1000
                    results[call.index] = ToolResult(
                        call=call, result=result, duration_ms=duration
                    )
                except Exception as exc:
                    duration = (time.monotonic() - t0) * 1000
                    results[call.index] = ToolResult(
                        call=call, error=exc, duration_ms=duration
                    )
                finally:
                    work_q.task_done()

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
        for t in threads:
            t.start()

        deadline = None if timeout is None else time.monotonic() + timeout
        for t in threads:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            t.join(timeout=remaining)

        # Fill in any calls that did not finish before the deadline so the
        # returned list always matches the input length and order.
        for i, (call, res) in enumerate(zip(calls, results)):
            if res is None:
                results[i] = ToolResult(
                    call=call,
                    error=TimeoutError(
                        f"Tool {call.name!r} did not finish within {timeout}s"
                    ),
                )

        return [r for r in results if r is not None]

    def run_one(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Convenience: run a single tool call."""
        results = self.run([ToolCall(name=tool_name, args=kwargs)])
        return results[0]

    @property
    def registered_tools(self) -> list[str]:
        return list(self._registry.keys())


def run_parallel(
    calls: list[dict[str, Any]],
    registry: dict[str, Callable[..., Any]],
    max_workers: int = 8,
) -> list[ToolResult]:
    """
    Convenience function: run a list of ``{"name": ..., "args": {...}}`` dicts.
    """
    runner = ParallelToolRunner(max_workers=max_workers)
    for name, fn in registry.items():
        runner.register(name, fn)
    tool_calls = [ToolCall(name=d["name"], args=d.get("args", {})) for d in calls]
    return runner.run(tool_calls)


__all__ = ["ParallelToolRunner", "ToolCall", "ToolResult", "run_parallel"]
