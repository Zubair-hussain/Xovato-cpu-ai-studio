# Reports

Shareable, self-contained HTML reports of the project's health. Open
[`index.html`](index.html) for the dashboard.

| File | What it is |
| --- | --- |
| `index.html` | Dashboard: pass count, coverage, links to everything. |
| `tests.html` | Full pytest run (pytest-html, self-contained). |
| `coverage-summary.html` | Per-module line coverage on one page. |
| `coverage/index.html` | Full line-by-line coverage report. |
| `benchmarks.html` | Measured CPU throughput + comparison charts. |
| `coverage.xml`, `junit.xml` | Machine-readable artefacts (for CI badges, etc.). |

## Regenerate

```bash
pip install -r requirements-dev.txt        # includes pytest-html
pytest -o addopts="" --cov=app \
  --cov-report=xml:docs/reports/coverage.xml \
  --cov-report=html:docs/reports/coverage \
  --html=docs/reports/tests.html --self-contained-html \
  --junitxml=docs/reports/junit.xml
python scripts/generate_reports.py
```

Everything here is generated from a real test run - nothing is hand-edited, and
there is nothing personal in it. Safe to publish (e.g. GitHub Pages).
