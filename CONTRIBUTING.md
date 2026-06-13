# Contributing to agent-parallel-tools

Thanks for your interest in contributing! This project lets you run multiple
agent tool calls concurrently while preserving result order, with zero runtime
dependencies on Python 3.10+.

## Getting started

1. Fork the repository and clone your fork.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Linux/macOS
   source .venv/bin/activate
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```

3. Install the package with its development extras (in editable mode):

   ```bash
   python -m pip install --upgrade pip
   pip install -e ".[dev]"
   ```

## Development workflow

- Create a feature branch off `main` for your change.
- Keep changes focused and small; one logical change per pull request.
- Add or update tests for any behavior you change.

## Linting and tests

The CI pipeline runs the same checks locally, so please run them before
pushing:

```bash
# Lint
ruff check src/ tests/

# Tests
pytest -v --tb=short
```

CI runs against Python 3.10, 3.11, 3.12, and 3.13. Make sure your change works
across supported versions where practical.

## Submitting changes

1. Ensure lint and tests pass locally.
2. Push your branch and open a pull request against `main`.
3. Describe the motivation and summarize the change in the PR description.
4. Be responsive to review feedback.

## Reporting issues

When filing an issue, please include:

- A clear description of the problem or feature request.
- Steps to reproduce (for bugs), including your Python version.
- Expected vs. actual behavior.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE) that covers this project.
