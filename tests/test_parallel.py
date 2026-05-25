"""Tests for agent_parallel_tools.ParallelTools."""

import asyncio
import threading
import time

import pytest

from agent_parallel_tools import ParallelTools, ToolCall, ToolResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def add(args: dict) -> int:
    return args["a"] + args["b"]


def multiply(args: dict) -> int:
    return args["a"] * args["b"]


def always_raises(args: dict) -> None:
    raise ValueError("boom")


async def async_double(args: dict) -> int:
    await asyncio.sleep(0)
    return args["x"] * 2


# ---------------------------------------------------------------------------
# register() + run() basics
# ---------------------------------------------------------------------------


def test_register_and_run_correct_args():
    tools = ParallelTools()
    received = {}

    def capture(args):
        received.update(args)
        return "ok"

    tools.register("capture", capture)
    tools.run([ToolCall(name="capture", args={"key": "value"})])
    assert received == {"key": "value"}


def test_register_returns_self_for_chaining():
    tools = ParallelTools()
    result = tools.register("add", add)
    assert result is tools


def test_chaining_registers_multiple_tools():
    tools = ParallelTools().register("add", add).register("multiply", multiply)
    assert tools.is_registered("add")
    assert tools.is_registered("multiply")


# ---------------------------------------------------------------------------
# run() ordering and results
# ---------------------------------------------------------------------------


def test_run_results_in_original_order():
    tools = ParallelTools().register("add", add).register("multiply", multiply)
    calls = [
        ToolCall(name="multiply", args={"a": 3, "b": 4}),
        ToolCall(name="add", args={"a": 10, "b": 5}),
        ToolCall(name="multiply", args={"a": 2, "b": 2}),
    ]
    results = tools.run(calls)
    assert results[0].result == 12
    assert results[1].result == 15
    assert results[2].result == 4


def test_run_success_result_populated():
    tools = ParallelTools().register("add", add)
    results = tools.run([ToolCall(name="add", args={"a": 7, "b": 3})])
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, ToolResult)
    assert r.result == 10
    assert r.error is None
    assert r.name == "add"
    assert r.args == {"a": 7, "b": 3}


def test_run_tool_error_captured_not_raised():
    tools = ParallelTools().register("boom", always_raises)
    results = tools.run([ToolCall(name="boom", args={})])
    assert len(results) == 1
    r = results[0]
    assert r.result is None
    assert r.error == "boom"


def test_run_unregistered_tool_error_not_raised():
    tools = ParallelTools()
    results = tools.run([ToolCall(name="ghost", args={})])
    assert len(results) == 1
    r = results[0]
    assert r.result is None
    assert "ghost" in r.error
    assert "not registered" in r.error


def test_run_empty_list_returns_empty():
    tools = ParallelTools()
    assert tools.run([]) == []


def test_run_duration_ms_non_negative():
    tools = ParallelTools().register("add", add)
    results = tools.run([ToolCall(name="add", args={"a": 1, "b": 1})])
    assert results[0].duration_ms >= 0.0


def test_run_unregistered_duration_zero():
    tools = ParallelTools()
    results = tools.run([ToolCall(name="missing", args={})])
    assert results[0].duration_ms == 0.0


# ---------------------------------------------------------------------------
# run_async() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_async_awaits_async_fn():
    tools = ParallelTools().register("double", async_double)
    results = await tools.run_async([ToolCall(name="double", args={"x": 6})])
    assert results[0].result == 12
    assert results[0].error is None


@pytest.mark.asyncio
async def test_run_async_sync_fn_works():
    tools = ParallelTools().register("add", add)
    results = await tools.run_async([ToolCall(name="add", args={"a": 3, "b": 4})])
    assert results[0].result == 7


@pytest.mark.asyncio
async def test_run_async_results_in_original_order():
    async def slow(args):
        await asyncio.sleep(0.05)
        return args["v"]

    async def fast(args):
        return args["v"]

    tools = ParallelTools().register("slow", slow).register("fast", fast)
    calls = [
        ToolCall(name="slow", args={"v": "first"}),
        ToolCall(name="fast", args={"v": "second"}),
        ToolCall(name="slow", args={"v": "third"}),
    ]
    results = await tools.run_async(calls)
    assert results[0].result == "first"
    assert results[1].result == "second"
    assert results[2].result == "third"


@pytest.mark.asyncio
async def test_run_async_errors_captured():
    tools = ParallelTools().register("boom", always_raises)
    results = await tools.run_async([ToolCall(name="boom", args={})])
    assert results[0].result is None
    assert results[0].error == "boom"


@pytest.mark.asyncio
async def test_run_async_unregistered_error():
    tools = ParallelTools()
    results = await tools.run_async([ToolCall(name="ghost", args={})])
    assert results[0].result is None
    assert "ghost" in results[0].error


@pytest.mark.asyncio
async def test_run_async_empty_list():
    tools = ParallelTools()
    assert await tools.run_async([]) == []


@pytest.mark.asyncio
async def test_run_async_duration_non_negative():
    tools = ParallelTools().register("add", add)
    results = await tools.run_async([ToolCall(name="add", args={"a": 1, "b": 1})])
    assert results[0].duration_ms >= 0.0


# ---------------------------------------------------------------------------
# run_single()
# ---------------------------------------------------------------------------


def test_run_single_success():
    tools = ParallelTools().register("add", add)
    r = tools.run_single(ToolCall(name="add", args={"a": 5, "b": 5}))
    assert isinstance(r, ToolResult)
    assert r.result == 10
    assert r.error is None


def test_run_single_error_captured():
    tools = ParallelTools().register("boom", always_raises)
    r = tools.run_single(ToolCall(name="boom", args={}))
    assert r.result is None
    assert r.error == "boom"


def test_run_single_unregistered():
    tools = ParallelTools()
    r = tools.run_single(ToolCall(name="ghost", args={}))
    assert r.result is None
    assert "ghost" in r.error


# ---------------------------------------------------------------------------
# is_registered() / registered_tools
# ---------------------------------------------------------------------------


def test_is_registered_true():
    tools = ParallelTools().register("add", add)
    assert tools.is_registered("add") is True


def test_is_registered_false():
    tools = ParallelTools()
    assert tools.is_registered("add") is False


def test_registered_tools_sorted():
    tools = ParallelTools().register("zzz", add).register("aaa", add).register("mmm", add)
    assert tools.registered_tools == ["aaa", "mmm", "zzz"]


def test_registered_tools_empty():
    tools = ParallelTools()
    assert tools.registered_tools == []


# ---------------------------------------------------------------------------
# max_workers concurrency limit
# ---------------------------------------------------------------------------


def test_run_max_workers_limits_concurrency():
    """Verify at most max_workers tasks run simultaneously."""
    max_workers = 2
    tools = ParallelTools(max_workers=max_workers)

    concurrent_peak = 0
    lock = threading.Lock()
    running = 0

    def slow_tool(args):
        nonlocal running, concurrent_peak
        with lock:
            running += 1
            if running > concurrent_peak:
                concurrent_peak = running
        time.sleep(0.05)
        with lock:
            running -= 1
        return "done"

    # Register under multiple names to enqueue 5 calls
    for i in range(5):
        tools.register(f"t{i}", slow_tool)

    calls = [ToolCall(name=f"t{i}", args={}) for i in range(5)]
    tools.run(calls)

    assert concurrent_peak <= max_workers


@pytest.mark.asyncio
async def test_run_async_max_workers_limits_concurrency():
    """Verify semaphore keeps concurrency within max_workers."""
    max_workers = 2
    tools = ParallelTools(max_workers=max_workers)

    concurrent_peak = 0
    running = 0

    async def slow_tool(args):
        nonlocal running, concurrent_peak
        running += 1
        if running > concurrent_peak:
            concurrent_peak = running
        await asyncio.sleep(0.05)
        running -= 1
        return "done"

    for i in range(5):
        tools.register(f"t{i}", slow_tool)

    calls = [ToolCall(name=f"t{i}", args={}) for i in range(5)]
    await tools.run_async(calls)

    assert concurrent_peak <= max_workers


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_run_multiple_errors_all_captured():
    """All failed calls are captured; none raises."""
    tools = (
        ParallelTools()
        .register("boom1", lambda args: (_ for _ in ()).throw(RuntimeError("err1")))
        .register("boom2", lambda args: (_ for _ in ()).throw(RuntimeError("err2")))
    )
    results = tools.run([
        ToolCall(name="boom1", args={}),
        ToolCall(name="boom2", args={}),
    ])
    assert results[0].error == "err1"
    assert results[1].error == "err2"


def test_run_mixed_success_and_error_order():
    """Mixed success/error calls preserve original order."""
    tools = ParallelTools().register("add", add).register("boom", always_raises)
    calls = [
        ToolCall(name="add", args={"a": 1, "b": 1}),
        ToolCall(name="boom", args={}),
        ToolCall(name="add", args={"a": 2, "b": 2}),
    ]
    results = tools.run(calls)
    assert results[0].result == 2
    assert results[0].error is None
    assert results[1].result is None
    assert results[1].error == "boom"
    assert results[2].result == 4
    assert results[2].error is None
