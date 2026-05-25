# agent-parallel-tools

[![PyPI](https://img.shields.io/pypi/v/agent-parallel-tools.svg)](https://pypi.org/project/agent-parallel-tools/)
[![Python](https://img.shields.io/pypi/pyversions/agent-parallel-tools.svg)](https://pypi.org/project/agent-parallel-tools/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Execute multiple LLM `tool_use` blocks in parallel with a concurrency limit. Results returned in original call order. Zero runtime dependencies.

## Why

When an LLM returns multiple `tool_use` blocks in one response, the naive approach runs them sequentially. This library runs them concurrently, bounded by `max_workers`, so a batch of five web-search calls takes the time of one instead of five.

Not the same as batching LLM requests together — this is about running the results of one LLM response in parallel.

## Install

```bash
pip install agent-parallel-tools
```

## Quick start

```python
from agent_parallel_tools import ParallelTools, ToolCall

tools = ParallelTools(max_workers=4)
tools.register("search", lambda args: web_search(args["query"]))
tools.register("lookup", lambda args: db_lookup(args["id"]))

# LLM returned these two tool_use blocks
results = tools.run([
    ToolCall(name="search", args={"query": "latest Python release"}),
    ToolCall(name="lookup", args={"id": "user-42"}),
])

for r in results:
    if r.error:
        print(f"{r.name} failed: {r.error}")
    else:
        print(f"{r.name} => {r.result}")
```

## Async

```python
async def fetch(args):
    async with session.get(args["url"]) as resp:
        return await resp.text()

tools = ParallelTools(max_workers=8)
tools.register("fetch", fetch)

results = await tools.run_async([
    ToolCall(name="fetch", args={"url": "https://example.com"}),
    ToolCall(name="fetch", args={"url": "https://example.org"}),
])
```

Sync functions also work in `run_async` — they are dispatched to the default executor automatically.

## API

### `ParallelTools(max_workers=5)`

| Method | Description |
|--------|-------------|
| `register(name, fn)` | Register a sync or async callable. Returns `self` for chaining. |
| `is_registered(name)` | Returns `True` if the tool is registered. |
| `registered_tools` | Sorted list of registered tool names (property). |
| `run(calls)` | Run calls in parallel via `ThreadPoolExecutor`. Returns results in original order. |
| `run_async(calls)` | Run calls concurrently via `asyncio`. Returns results in original order. |
| `run_single(call)` | Convenience wrapper for a single call. |

### `ToolCall`

```python
@dataclass
class ToolCall:
    name: str
    args: dict
```

### `ToolResult`

```python
@dataclass
class ToolResult:
    name: str
    args: dict
    result: object      # None on error
    error: str | None   # None on success
    duration_ms: float
```

Errors are captured per-call — a failing tool does not raise from `run()` or `run_async()`.

## License

MIT
