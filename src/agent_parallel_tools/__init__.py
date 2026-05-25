"""agent-parallel-tools - execute multiple LLM tool_use blocks in parallel.

When an LLM returns multiple tool_use blocks, run them concurrently instead of
sequentially. Results are returned in the same order as the input calls.

    from agent_parallel_tools import ParallelTools, ToolCall

    tools = ParallelTools(max_workers=4)
    tools.register("add", lambda args: args["a"] + args["b"])
    tools.register("multiply", lambda args: args["a"] * args["b"])

    results = tools.run([
        ToolCall(name="add", args={"a": 1, "b": 2}),
        ToolCall(name="multiply", args={"a": 3, "b": 4}),
    ])
    # results[0].result == 3, results[1].result == 12

Async usage::

    results = await tools.run_async([
        ToolCall(name="add", args={"a": 1, "b": 2}),
    ])

Errors are captured per-call rather than raising, keeping other calls' results
intact::

    result = results[0]
    if result.error:
        print(f"Tool failed: {result.error}")
    else:
        print(f"Tool returned: {result.result}")
"""

import asyncio
import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class ToolCall:
    """A single tool invocation from an LLM tool_use block."""

    name: str
    args: dict


@dataclass
class ToolResult:
    """The result of a single tool invocation."""

    name: str
    args: dict
    result: object  # None if error
    error: str | None  # None if success
    duration_ms: float


class ToolNotRegisteredError(Exception):
    """Raised when a tool name is looked up but not registered."""


class ParallelTools:
    """Run multiple tool calls concurrently, returning results in original order.

    Args:
        max_workers: Maximum number of tool calls to run simultaneously.
    """

    def __init__(self, max_workers: int = 5) -> None:
        self._max_workers = max_workers
        self._registry: dict[str, object] = {}

    def register(self, name: str, fn) -> "ParallelTools":
        """Register a tool function (sync or async). Returns self for chaining."""
        self._registry[name] = fn
        return self

    def is_registered(self, name: str) -> bool:
        """Return True if a tool with the given name is registered."""
        return name in self._registry

    @property
    def registered_tools(self) -> list[str]:
        """Sorted list of registered tool names."""
        return sorted(self._registry.keys())

    def _invoke_sync(self, call: ToolCall) -> ToolResult:
        """Invoke a single (sync) tool call, capturing errors."""
        start = time.monotonic()
        fn = self._registry.get(call.name)
        if fn is None:
            return ToolResult(
                name=call.name,
                args=call.args,
                result=None,
                error=f"Tool not registered: {call.name}",
                duration_ms=0.0,
            )
        try:
            result = fn(call.args)
            return ToolResult(
                name=call.name,
                args=call.args,
                result=result,
                error=None,
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            return ToolResult(
                name=call.name,
                args=call.args,
                result=None,
                error=str(exc),
                duration_ms=(time.monotonic() - start) * 1000,
            )

    def run(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Run tool calls in parallel using a thread pool.

        Results are returned in the same order as *calls*, regardless of
        completion order. Errors are captured per-call; no exception is raised
        from this method.

        Args:
            calls: List of ToolCall objects to execute.

        Returns:
            List of ToolResult objects in the same order as *calls*.
        """
        if not calls:
            return []

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [executor.submit(self._invoke_sync, call) for call in calls]
            return [f.result() for f in futures]

    async def run_async(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Run tool calls concurrently using asyncio.

        Coroutine functions are awaited directly; synchronous functions are
        dispatched to the default executor. A semaphore limits concurrency to
        *max_workers*. Results are returned in the same order as *calls*.

        Args:
            calls: List of ToolCall objects to execute.

        Returns:
            List of ToolResult objects in the same order as *calls*.
        """
        if not calls:
            return []

        semaphore = asyncio.Semaphore(self._max_workers)
        loop = asyncio.get_event_loop()

        async def _run_one(call: ToolCall) -> ToolResult:
            async with semaphore:
                start = time.monotonic()
                fn = self._registry.get(call.name)
                if fn is None:
                    return ToolResult(
                        name=call.name,
                        args=call.args,
                        result=None,
                        error=f"Tool not registered: {call.name}",
                        duration_ms=0.0,
                    )
                try:
                    if inspect.iscoroutinefunction(fn):
                        result = await fn(call.args)
                    else:
                        result = await loop.run_in_executor(None, fn, call.args)
                    return ToolResult(
                        name=call.name,
                        args=call.args,
                        result=result,
                        error=None,
                        duration_ms=(time.monotonic() - start) * 1000,
                    )
                except Exception as exc:
                    return ToolResult(
                        name=call.name,
                        args=call.args,
                        result=None,
                        error=str(exc),
                        duration_ms=(time.monotonic() - start) * 1000,
                    )

        tasks = [asyncio.create_task(_run_one(call)) for call in calls]
        return list(await asyncio.gather(*tasks))

    def run_single(self, call: ToolCall) -> ToolResult:
        """Convenience method to run a single tool call synchronously."""
        return self._invoke_sync(call)


__version__ = "0.1.0"

__all__ = [
    "ParallelTools",
    "ToolCall",
    "ToolNotRegisteredError",
    "ToolResult",
    "__version__",
]
