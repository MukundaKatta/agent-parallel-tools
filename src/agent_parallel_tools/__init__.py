"""
agent-parallel-tools: Run multiple tool calls in parallel, results in original order.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
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

    def run(self, calls: list[ToolCall], timeout: Optional[float] = None) -> list[ToolResult]:
        """
        Execute all calls concurrently. Returns results in original order.
        Raises KeyError for unregistered tools (before starting any threads).
        """
        for call in calls:
            call.index = calls.index(call)
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
            import time
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
                    results[call.index] = ToolResult(call=call, result=result, duration_ms=duration)
                except Exception as exc:
                    duration = (time.monotonic() - t0) * 1000
                    results[call.index] = ToolResult(call=call, error=exc, duration_ms=duration)
                finally:
                    work_q.task_done()

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=timeout)

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
