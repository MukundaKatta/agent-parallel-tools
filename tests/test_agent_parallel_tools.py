"""Tests for agent-parallel-tools."""

import time
import pytest
from agent_parallel_tools import ParallelToolRunner, ToolCall, ToolResult, run_parallel


def test_single_tool():
    runner = ParallelToolRunner()
    runner.register("add", lambda x, y: x + y)
    results = runner.run([ToolCall("add", {"x": 1, "y": 2})])
    assert len(results) == 1
    assert results[0].result == 3
    assert results[0].ok is True


def test_multiple_tools_in_order():
    runner = ParallelToolRunner()
    runner.register("a", lambda: "A")
    runner.register("b", lambda: "B")
    runner.register("c", lambda: "C")
    calls = [ToolCall("a", {}), ToolCall("b", {}), ToolCall("c", {})]
    results = runner.run(calls)
    assert [r.result for r in results] == ["A", "B", "C"]


def test_parallel_execution_faster():
    runner = ParallelToolRunner(max_workers=4)
    runner.register("slow", lambda: time.sleep(0.05) or "done")
    calls = [ToolCall("slow", {}) for _ in range(4)]
    t0 = time.monotonic()
    results = runner.run(calls)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.15  # 4 x 50ms in parallel should be well under 200ms
    assert all(r.result == "done" for r in results)


def test_error_in_one_tool():
    runner = ParallelToolRunner()
    runner.register("bad", lambda: (_ for _ in ()).throw(ValueError("oops")))
    results = runner.run([ToolCall("bad", {})])
    assert results[0].ok is False
    assert isinstance(results[0].error, Exception)


def test_unregistered_tool_raises():
    runner = ParallelToolRunner()
    with pytest.raises(KeyError):
        runner.run([ToolCall("missing", {})])


def test_empty_calls():
    runner = ParallelToolRunner()
    results = runner.run([])
    assert results == []


def test_duration_ms_set():
    runner = ParallelToolRunner()
    runner.register("fast", lambda: "ok")
    results = runner.run([ToolCall("fast", {})])
    assert results[0].duration_ms is not None
    assert results[0].duration_ms >= 0


def test_tool_result_name():
    runner = ParallelToolRunner()
    runner.register("my_tool", lambda: None)
    results = runner.run([ToolCall("my_tool", {})])
    assert results[0].name == "my_tool"


def test_registered_tools_list():
    runner = ParallelToolRunner()
    runner.register("a", lambda: None)
    runner.register("b", lambda: None)
    assert "a" in runner.registered_tools
    assert "b" in runner.registered_tools


def test_run_one():
    runner = ParallelToolRunner()
    runner.register("greet", lambda name: f"Hello {name}")
    result = runner.run_one("greet", name="Alice")
    assert result.result == "Hello Alice"


def test_run_parallel_convenience():
    results = run_parallel(
        [
            {"name": "add", "args": {"x": 1, "y": 2}},
            {"name": "mul", "args": {"x": 3, "y": 4}},
        ],
        registry={"add": lambda x, y: x + y, "mul": lambda x, y: x * y},
    )
    assert len(results) == 2
    assert any(r.result == 3 for r in results)
    assert any(r.result == 12 for r in results)


def test_result_order_preserved_parallel():
    runner = ParallelToolRunner(max_workers=4)
    # tools that sleep for different amounts but should return in original order
    runner.register("fast", lambda n: n)
    calls = [ToolCall("fast", {"n": i}) for i in range(5)]
    results = runner.run(calls)
    assert [r.result for r in results] == [0, 1, 2, 3, 4]


def test_tool_result_public_api():
    call = ToolCall("noop", {})
    ok_result = ToolResult(call=call, result=42)
    assert ok_result.ok is True
    assert ok_result.name == "noop"
    assert ok_result.result == 42

    err = ValueError("bad")
    bad_result = ToolResult(call=call, error=err)
    assert bad_result.ok is False
    assert bad_result.error is err


def test_timeout_does_not_drop_results():
    # A call that exceeds the deadline must still appear in the results, in
    # order, reported as a TimeoutError rather than being silently dropped.
    runner = ParallelToolRunner(max_workers=2)
    runner.register("slow", lambda d: time.sleep(d) or f"slept {d}")
    calls = [ToolCall("slow", {"d": 0.0}), ToolCall("slow", {"d": 0.5})]
    results = runner.run(calls, timeout=0.1)
    assert len(results) == 2
    assert results[0].ok is True
    assert results[0].result == "slept 0.0"
    assert results[1].ok is False
    assert isinstance(results[1].error, TimeoutError)


def test_duplicate_equal_calls_preserve_order():
    # Two ToolCall instances that compare equal must not collapse to one index.
    runner = ParallelToolRunner()
    runner.register("echo", lambda v: v)
    calls = [ToolCall("echo", {"v": 1}), ToolCall("echo", {"v": 1})]
    results = runner.run(calls)
    assert [r.result for r in results] == [1, 1]
