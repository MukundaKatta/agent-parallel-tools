"""Tests for agent-parallel-tools (standard-library unittest only).

Run with::

    python3 -m unittest discover -s tests
"""
import threading
import time
import unittest

from agent_parallel_tools import (
    ParallelToolRunner,
    ToolCall,
    ToolResult,
    run_parallel,
)


class SingleAndOrderTests(unittest.TestCase):
    def test_single_tool(self):
        runner = ParallelToolRunner()
        runner.register("add", lambda x, y: x + y)
        results = runner.run([ToolCall("add", {"x": 1, "y": 2})])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].result, 3)
        self.assertTrue(results[0].ok)

    def test_multiple_tools_in_order(self):
        runner = ParallelToolRunner()
        runner.register("a", lambda: "A")
        runner.register("b", lambda: "B")
        runner.register("c", lambda: "C")
        calls = [ToolCall("a", {}), ToolCall("b", {}), ToolCall("c", {})]
        results = runner.run(calls)
        self.assertEqual([r.result for r in results], ["A", "B", "C"])

    def test_result_order_preserved_parallel(self):
        runner = ParallelToolRunner(max_workers=4)
        runner.register("echo", lambda n: n)
        calls = [ToolCall("echo", {"n": i}) for i in range(5)]
        results = runner.run(calls)
        self.assertEqual([r.result for r in results], [0, 1, 2, 3, 4])

    def test_order_preserved_despite_uneven_durations(self):
        # The first call sleeps longest; results must still come back in
        # input order, not completion order.
        runner = ParallelToolRunner(max_workers=3)
        runner.register("sleep_then", lambda d, v: (time.sleep(d), v)[1])
        calls = [
            ToolCall("sleep_then", {"d": 0.15, "v": "first"}),
            ToolCall("sleep_then", {"d": 0.05, "v": "second"}),
            ToolCall("sleep_then", {"d": 0.0, "v": "third"}),
        ]
        results = runner.run(calls)
        self.assertEqual(
            [r.result for r in results], ["first", "second", "third"]
        )

    def test_identical_calls_each_get_their_own_result(self):
        # ToolCall is a dataclass, so identical calls compare equal. The runner
        # must not collapse them onto a single result slot.
        runner = ParallelToolRunner(max_workers=4)
        runner.register("const", lambda: "x")
        calls = [ToolCall("const", {}) for _ in range(4)]
        results = runner.run(calls)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(r.result == "x" for r in results))
        self.assertEqual([c.index for c in calls], [0, 1, 2, 3])


class ConcurrencyTests(unittest.TestCase):
    def test_parallel_execution_faster(self):
        runner = ParallelToolRunner(max_workers=4)
        runner.register("slow", lambda: time.sleep(0.05) or "done")
        calls = [ToolCall("slow", {}) for _ in range(4)]
        t0 = time.monotonic()
        results = runner.run(calls)
        elapsed = time.monotonic() - t0
        # 4 x 50ms run concurrently; serial would be ~200ms.
        self.assertLess(elapsed, 0.15)
        self.assertTrue(all(r.result == "done" for r in results))

    def test_workers_capped_at_num_calls(self):
        # With more workers than calls, only as many threads as calls should
        # actually pull work. We assert correctness rather than thread count.
        runner = ParallelToolRunner(max_workers=100)
        runner.register("echo", lambda n: n)
        calls = [ToolCall("echo", {"n": i}) for i in range(3)]
        results = runner.run(calls)
        self.assertEqual([r.result for r in results], [0, 1, 2])

    def test_concurrency_actually_overlaps(self):
        # Use a barrier sized to the worker count: if all workers run at once,
        # the barrier releases; otherwise this would deadlock/time out.
        runner = ParallelToolRunner(max_workers=3)
        barrier = threading.Barrier(3, timeout=2.0)
        runner.register("wait", lambda: barrier.wait() or "ok")
        calls = [ToolCall("wait", {}) for _ in range(3)]
        results = runner.run(calls)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.ok for r in results))


class ErrorTests(unittest.TestCase):
    def test_error_in_one_tool_is_captured(self):
        def boom():
            raise ValueError("oops")

        runner = ParallelToolRunner()
        runner.register("bad", boom)
        results = runner.run([ToolCall("bad", {})])
        self.assertFalse(results[0].ok)
        self.assertIsInstance(results[0].error, ValueError)
        self.assertIsNone(results[0].result)

    def test_error_does_not_block_other_calls(self):
        def boom():
            raise RuntimeError("nope")

        runner = ParallelToolRunner(max_workers=2)
        runner.register("bad", boom)
        runner.register("good", lambda: "fine")
        calls = [ToolCall("bad", {}), ToolCall("good", {})]
        results = runner.run(calls)
        self.assertFalse(results[0].ok)
        self.assertTrue(results[1].ok)
        self.assertEqual(results[1].result, "fine")

    def test_unregistered_tool_raises_keyerror(self):
        runner = ParallelToolRunner()
        with self.assertRaises(KeyError):
            runner.run([ToolCall("missing", {})])

    def test_unregistered_tool_aborts_before_running_anything(self):
        # If any call is invalid, nothing should execute (all-or-nothing
        # validation). Track whether the valid tool ever ran.
        ran = []
        runner = ParallelToolRunner()
        runner.register("good", lambda: ran.append(1))
        with self.assertRaises(KeyError):
            runner.run([ToolCall("good", {}), ToolCall("missing", {})])
        self.assertEqual(ran, [])

    def test_invalid_max_workers_rejected(self):
        with self.assertRaises(ValueError):
            ParallelToolRunner(max_workers=0)
        with self.assertRaises(ValueError):
            ParallelToolRunner(max_workers=-1)


class TimeoutTests(unittest.TestCase):
    def test_timeout_yields_timeout_result_but_keeps_order_and_length(self):
        runner = ParallelToolRunner(max_workers=2)
        runner.register("fast", lambda: "fast")
        runner.register("slow", lambda: time.sleep(0.5) or "slow")
        calls = [ToolCall("fast", {}), ToolCall("slow", {})]
        results = runner.run(calls, timeout=0.05)
        # The contract: same length and index alignment as the input calls.
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].name, "fast")
        self.assertEqual(results[0].result, "fast")
        self.assertEqual(results[1].name, "slow")
        self.assertFalse(results[1].ok)
        self.assertIsInstance(results[1].error, TimeoutError)

    def test_no_timeout_completes_normally(self):
        runner = ParallelToolRunner(max_workers=2)
        runner.register("work", lambda: time.sleep(0.02) or "done")
        calls = [ToolCall("work", {}) for _ in range(2)]
        results = runner.run(calls, timeout=5.0)
        self.assertTrue(all(r.ok and r.result == "done" for r in results))


class MetadataTests(unittest.TestCase):
    def test_empty_calls_returns_empty_list(self):
        runner = ParallelToolRunner()
        self.assertEqual(runner.run([]), [])

    def test_duration_ms_is_set_and_nonnegative(self):
        runner = ParallelToolRunner()
        runner.register("fast", lambda: "ok")
        results = runner.run([ToolCall("fast", {})])
        self.assertIsNotNone(results[0].duration_ms)
        self.assertGreaterEqual(results[0].duration_ms, 0)

    def test_tool_result_name_property(self):
        runner = ParallelToolRunner()
        runner.register("my_tool", lambda: None)
        results = runner.run([ToolCall("my_tool", {})])
        self.assertEqual(results[0].name, "my_tool")

    def test_ok_property(self):
        ok = ToolResult(call=ToolCall("t", {}), result=1)
        bad = ToolResult(call=ToolCall("t", {}), error=ValueError())
        self.assertTrue(ok.ok)
        self.assertFalse(bad.ok)

    def test_registered_tools_list(self):
        runner = ParallelToolRunner()
        runner.register("a", lambda: None)
        runner.register("b", lambda: None)
        self.assertIn("a", runner.registered_tools)
        self.assertIn("b", runner.registered_tools)

    def test_register_is_chainable(self):
        runner = ParallelToolRunner()
        returned = runner.register("a", lambda: None).register("b", lambda: None)
        self.assertIs(returned, runner)
        self.assertEqual(set(runner.registered_tools), {"a", "b"})


class ConvenienceTests(unittest.TestCase):
    def test_run_one(self):
        runner = ParallelToolRunner()
        runner.register("greet", lambda name: f"Hello {name}")
        result = runner.run_one("greet", name="Alice")
        self.assertEqual(result.result, "Hello Alice")

    def test_run_parallel_convenience(self):
        results = run_parallel(
            [
                {"name": "add", "args": {"x": 1, "y": 2}},
                {"name": "mul", "args": {"x": 3, "y": 4}},
            ],
            registry={
                "add": lambda x, y: x + y,
                "mul": lambda x, y: x * y,
            },
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].result, 3)
        self.assertEqual(results[1].result, 12)

    def test_run_parallel_defaults_args_to_empty(self):
        results = run_parallel(
            [{"name": "ping"}],
            registry={"ping": lambda: "pong"},
        )
        self.assertEqual(results[0].result, "pong")


if __name__ == "__main__":
    unittest.main()
