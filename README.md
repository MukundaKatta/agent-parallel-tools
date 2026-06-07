# agent-parallel-tools

[![CI](https://github.com/MukundaKatta/agent-parallel-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/agent-parallel-tools/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Run multiple agent tool calls **concurrently** and get the results back **in the
original call order**. Zero runtime dependencies, pure standard library,
Python 3.10+.

When an LLM agent emits several tool calls in a single turn, you usually want to
execute them at the same time (they're typically I/O-bound: HTTP requests, file
reads, subprocess calls) but hand the results back to the model in the exact
order it asked for them. `agent-parallel-tools` does precisely that, with a tiny,
explicit API.

## Why

- **Order-preserving.** `results[i]` always corresponds to `calls[i]`, no matter
  which call finishes first.
- **Errors are values, not crashes.** A failing tool produces a `ToolResult`
  with `ok == False` and the captured exception, so one bad call never sinks the
  whole batch.
- **Bounded concurrency.** A simple thread pool capped by `max_workers`.
- **Per-batch timeout.** Cap the wall-clock time for an entire batch; calls that
  don't finish come back as `TimeoutError` results without breaking ordering.
- **Zero dependencies.** Nothing to install beyond the standard library.

## Install

From source (PyPI release pending):

```bash
pip install git+https://github.com/MukundaKatta/agent-parallel-tools.git
```

Or clone and install in editable mode:

```bash
git clone https://github.com/MukundaKatta/agent-parallel-tools.git
cd agent-parallel-tools
pip install -e .
```

## Quick start

```python
import time
from agent_parallel_tools import ParallelToolRunner, ToolCall


def search(query: str) -> str:
    time.sleep(0.1)            # pretend this is an API call
    return f"results for {query}"


def fetch(url: str) -> int:
    time.sleep(0.1)            # pretend this is an HTTP GET
    return len(url)


runner = ParallelToolRunner(max_workers=8)
runner.register("search", search).register("fetch", fetch)

calls = [
    ToolCall("search", {"query": "python"}),
    ToolCall("fetch", {"url": "https://example.com"}),
]

results = runner.run(calls)           # both run concurrently, ~0.1s total

for r in results:                     # iterated in the original call order
    if r.ok:
        print(f"{r.name}: {r.result}  ({r.duration_ms:.1f} ms)")
    else:
        print(f"{r.name} failed: {r.error!r}")
```

Output:

```
search: results for python  (100.x ms)
fetch: 19  (100.x ms)
```

### Handling errors

A tool that raises does not stop the batch; the exception is captured on the
result:

```python
def flaky():
    raise ValueError("boom")

runner = ParallelToolRunner()
runner.register("flaky", flaky).register("ok", lambda: "fine")

results = runner.run([ToolCall("flaky", {}), ToolCall("ok", {})])
results[0].ok        # False
results[0].error     # ValueError('boom')
results[1].result    # 'fine'
```

### Per-batch timeout

Give the whole batch a wall-clock budget. Calls that don't finish in time come
back as `TimeoutError` results, and ordering/length are preserved:

```python
runner.run(calls, timeout=2.0)   # at most ~2s total
```

### Calls as plain dicts

If your tool calls arrive as dictionaries (e.g. decoded from a model response),
use the `run_parallel` convenience function:

```python
from agent_parallel_tools import run_parallel

results = run_parallel(
    calls=[
        {"name": "add", "args": {"x": 1, "y": 2}},
        {"name": "mul", "args": {"x": 3, "y": 4}},
    ],
    registry={
        "add": lambda x, y: x + y,
        "mul": lambda x, y: x * y,
    },
    max_workers=4,
)
# results[0].result == 3, results[1].result == 12
```

## API

### `ToolCall(name, args, index=0)`

A single tool invocation request.

| Field   | Type             | Description                                                        |
| ------- | ---------------- | ------------------------------------------------------------------ |
| `name`  | `str`            | Registered name of the tool to invoke.                             |
| `args`  | `dict[str, Any]` | Keyword arguments, passed as `fn(**args)`.                         |
| `index` | `int`            | Position within the batch; set automatically by `run`. Leave as 0. |

### `ToolResult`

The outcome of one call. Exactly one of `result` / `error` is meaningful.

| Member        | Type                       | Description                                  |
| ------------- | -------------------------- | -------------------------------------------- |
| `call`        | `ToolCall`                 | The originating call.                        |
| `result`      | `Any`                      | Return value, or `None` on failure.          |
| `error`       | `BaseException` \| `None`  | Exception raised by the tool, or `None`.     |
| `duration_ms` | `float` \| `None`          | Wall-clock execution time in milliseconds.   |
| `ok`          | `bool` (property)          | `True` if the tool completed without raising.|
| `name`        | `str` (property)           | The tool name (shortcut for `call.name`).    |

### `ParallelToolRunner(max_workers=8)`

Registers tools and runs batches. Raises `ValueError` if `max_workers < 1`.

- **`register(name, fn) -> ParallelToolRunner`** — register a callable under
  `name`. Returns `self` for chaining.
- **`run(calls, timeout=None) -> list[ToolResult]`** — execute all calls
  concurrently and return results in input order. Raises `KeyError` (before
  starting any thread) if any call names an unregistered tool. `timeout` is an
  overall budget in seconds for the whole batch.
- **`run_one(tool_name, **kwargs) -> ToolResult`** — run a single call.
- **`registered_tools -> list[str]`** (property) — names of registered tools.

### `run_parallel(calls, registry, max_workers=8, timeout=None) -> list[ToolResult]`

Convenience wrapper: builds a runner, registers everything in `registry`, and
runs a list of `{"name": ..., "args": {...}}` dicts (`"args"` defaults to `{}`).

## Concurrency notes

Work runs in OS threads, so this is ideal for **I/O-bound** tools. CPU-bound
tools are limited by the GIL and will not see real speedup — reach for a process
pool there instead.

## Development

Run the test suite with the standard library (no third-party test deps needed):

```bash
python -m unittest discover -s tests -v
```

## License

[MIT](LICENSE)
