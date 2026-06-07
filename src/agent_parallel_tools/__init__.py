"""agent-parallel-tools: run multiple agent tool calls concurrently.

This module lets you fan out a batch of tool invocations across a thread
pool and collect the results back **in the original call order**, which is
exactly what an LLM agent loop needs when a model emits several tool calls
in a single turn.

The public surface is intentionally small:

* :class:`ToolCall` -- a single ``(name, args)`` invocation request.
* :class:`ToolResult` -- the outcome of one call (result *or* error, plus timing).
* :class:`ParallelToolRunner` -- registers tools and runs batches.
* :func:`run_parallel` -- a one-shot convenience wrapper.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

__version__ = "0.1.0"


@dataclass
class ToolCall:
    """A single tool invocation request.

    Attributes:
        name: The registered name of the tool to invoke.
        args: Keyword arguments passed to the tool function as ``fn(**args)``.
        index: Position of this call within its batch. Set automatically by
            :meth:`ParallelToolRunner.run`; callers normally leave it at ``0``.
    """

    name: str
    args: dict[str, Any]
    index: int = 0


@dataclass
class ToolResult:
    """The outcome of a single :class:`ToolCall`.

    Exactly one of :attr:`result` / :attr:`error` is meaningful: if the tool
    raised, :attr:`error` holds the exception and :attr:`result` is ``None``;
    otherwise :attr:`result` holds the return value and :attr:`error` is
    ``None``. Use :attr:`ok` to branch on success.

    Attributes:
        call: The originating :class:`ToolCall`.
        result: The tool's return value, or ``None`` on failure.
        error: The exception raised by the tool, or ``None`` on success.
        duration_ms: Wall-clock execution time in milliseconds.
    """

    call: ToolCall
    result: Any = None
    error: Optional[BaseException] = None
    duration_ms: Optional[float] = None

    @property
    def ok(self) -> bool:
        """``True`` if the tool completed without raising."""
        return self.error is None

    @property
    def name(self) -> str:
        """The name of the tool that produced this result."""
        return self.call.name


class ParallelToolRunner:
    """Execute a batch of tool calls concurrently, preserving call order.

    Tools are plain callables registered under a name. When :meth:`run` is
    given a list of :class:`ToolCall` objects it dispatches them across a
    bounded thread pool and returns a list of :class:`ToolResult` objects
    whose order matches the input list exactly -- ``results[i]`` always
    corresponds to ``calls[i]``, regardless of which call finished first.

    Because the work runs in threads, this is best suited to I/O-bound tools
    (HTTP requests, file reads, subprocess calls) rather than CPU-bound work,
    which is limited by the GIL.

    Example::

        runner = ParallelToolRunner(max_workers=8)
        runner.register("search", search_fn)
        runner.register("fetch", fetch_fn)

        calls = [
            ToolCall("search", {"query": "python"}),
            ToolCall("fetch", {"url": "https://example.com"}),
        ]
        results = runner.run(calls)
        # results[0].name == "search", results[1].name == "fetch"

    Args:
        max_workers: Maximum number of worker threads. The actual number used
            is ``min(max_workers, len(calls))``. Must be at least 1.

    Raises:
        ValueError: If ``max_workers`` is less than 1.
    """

    def __init__(self, max_workers: int = 8) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._max_workers = max_workers
        self._registry: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> "ParallelToolRunner":
        """Register a callable under ``name``.

        Returns ``self`` so registrations can be chained::

            runner.register("a", fn_a).register("b", fn_b)

        Args:
            name: The name callers will reference in :class:`ToolCall`.
            fn: The callable to invoke as ``fn(**call.args)``.

        Returns:
            This runner, to allow chaining.
        """
        self._registry[name] = fn
        return self

    def run(
        self, calls: list[ToolCall], timeout: Optional[float] = None
    ) -> list[ToolResult]:
        """Execute all calls concurrently and return results in order.

        Validation of tool names happens up front: if any call references an
        unregistered tool, :class:`KeyError` is raised before any thread is
        started, so a typo never partially executes a batch.

        The returned list always has the same length as ``calls`` and is
        index-aligned with it (``results[i]`` is the result of ``calls[i]``).

        Args:
            calls: The batch of tool calls to execute.
            timeout: Optional overall wall-clock budget, in seconds, for the
                whole batch. Calls that have not finished when the budget
                elapses yield a :class:`ToolResult` whose ``error`` is a
                :class:`TimeoutError`; ordering and length are still preserved.

        Returns:
            One :class:`ToolResult` per input call, in the original order.

        Raises:
            KeyError: If any call names a tool that is not registered.
        """
        if not calls:
            return []

        for call in calls:
            if call.name not in self._registry:
                raise KeyError(f"Tool not registered: {call.name!r}")

        results: list[Optional[ToolResult]] = [None] * len(calls)
        workers = min(self._max_workers, len(calls))
        work_q: "queue.Queue[ToolCall]" = queue.Queue()
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

        threads = [
            threading.Thread(target=worker, daemon=True) for _ in range(workers)
        ]
        for t in threads:
            t.start()

        # Join against a single shared deadline so that ``timeout`` is an
        # overall budget for the batch rather than a per-thread budget.
        deadline = None if timeout is None else time.monotonic() + timeout
        for t in threads:
            if deadline is None:
                t.join()
            else:
                remaining = deadline - time.monotonic()
                t.join(timeout=max(0.0, remaining))

        # Any slot still unfilled means that call did not finish within the
        # timeout. Fill it with a TimeoutError result so the returned list
        # stays the same length as ``calls`` and remains index-aligned.
        for call in calls:
            if results[call.index] is None:
                results[call.index] = ToolResult(
                    call=call,
                    error=TimeoutError(
                        f"Tool {call.name!r} did not complete within "
                        f"{timeout}s"
                    ),
                )

        return [r for r in results if r is not None]

    def run_one(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Run a single tool call and return its :class:`ToolResult`.

        Args:
            tool_name: The registered name of the tool to invoke.
            **kwargs: Keyword arguments forwarded to the tool.

        Returns:
            The :class:`ToolResult` for the single call.
        """
        results = self.run([ToolCall(name=tool_name, args=kwargs)])
        return results[0]

    @property
    def registered_tools(self) -> list[str]:
        """The names of all currently registered tools."""
        return list(self._registry.keys())


def run_parallel(
    calls: list[dict[str, Any]],
    registry: dict[str, Callable[..., Any]],
    max_workers: int = 8,
    timeout: Optional[float] = None,
) -> list[ToolResult]:
    """Run a list of ``{"name": ..., "args": {...}}`` dicts in parallel.

    A thin convenience wrapper that builds a :class:`ParallelToolRunner`,
    registers everything in ``registry``, and runs the batch. Handy when the
    tool calls arrive as plain dictionaries (e.g. decoded from an LLM
    response).

    Args:
        calls: A list of dicts, each with a ``"name"`` key and an optional
            ``"args"`` dict (defaults to ``{}`` when omitted).
        registry: Mapping of tool name to callable.
        max_workers: Maximum number of worker threads.
        timeout: Optional overall wall-clock budget, in seconds (see
            :meth:`ParallelToolRunner.run`).

    Returns:
        One :class:`ToolResult` per entry in ``calls``, in order.
    """
    runner = ParallelToolRunner(max_workers=max_workers)
    for name, fn in registry.items():
        runner.register(name, fn)
    tool_calls = [ToolCall(name=d["name"], args=d.get("args", {})) for d in calls]
    return runner.run(tool_calls, timeout=timeout)


__all__ = [
    "ParallelToolRunner",
    "ToolCall",
    "ToolResult",
    "run_parallel",
    "__version__",
]
