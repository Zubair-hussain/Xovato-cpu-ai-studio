# Contributing

Thanks for helping improve Xovato. This backend is intentionally CPU-first and
dependency-light; please keep changes in that spirit.

## Setup

Requires **Python 3.11+** (3.12 recommended).

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Run the API

```bash
uvicorn app.main:app --reload --port 8000
# docs at http://localhost:8000/docs
```

## Tests & coverage

```bash
pytest                 # runs the suite with coverage (see pytest.ini)
# HTML report is written to htmlcov/index.html
```

New code should come with tests. The heavy runtime paths (torch/OmniVoice,
ffmpeg, ONNX models, Supabase network) are not exercised in CI, so keep pure
logic separable and unit-testable - that is why, for example, caption grouping
and clip validation live in functions that do not need ffmpeg.

## Benchmarks

```bash
python scripts/benchmark.py         # measures on your machine
python scripts/generate_charts.py   # rewrites docs/assets/*.png
```

## Conventions

- Match the surrounding style: type hints, module docstrings that explain *why*,
  and comments only where the reason is not obvious from the code.
- No new GPU-only dependencies in `requirements.txt`. GPU acceleration belongs in
  the Colab notebook patches (see `docs/colab.md`), not the deploy package.
- Do not weaken the safety defaults: consent-gated cloning, upload limits, the
  `/files` allowlist, and the circuit breakers.
- Run `pytest` before opening a PR.
