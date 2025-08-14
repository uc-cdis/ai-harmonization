# AI Harmonization

This contains code and related artifacts for powering an AI-assisted data model harmonization tool. It also contains the infrastructure for abstracting approaches and benchmarking them.

## Setup

Python 3.12 is recommended, newer versions may also work.

This uses [uv](https://docs.astral.sh/uv/) (faster than `pip` and `poetry`, easier to work with).

```
uv sync
uv pip install -e .
```

## Tests

```
uv run pytest tests/
```

> Note: Tests are fairly minimal at the moment
