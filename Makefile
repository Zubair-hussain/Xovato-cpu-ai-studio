# Xovato CPU backend - common tasks
.PHONY: help install dev gpu test cov run bench charts clean

help:
	@echo "install  Install runtime dependencies (CPU)"
	@echo "dev      Install dev + test dependencies"
	@echo "gpu      Install optional NVIDIA GPU acceleration (see docs/gpu.md)"
	@echo "test     Run the test suite with coverage"
	@echo "cov      Open the HTML coverage report path"
	@echo "run      Start the API on :8000 with reload"
	@echo "bench    Measure CPU throughput -> docs/assets/benchmark_data.json"
	@echo "charts   Regenerate benchmark PNGs from the measured data"
	@echo "clean    Remove caches and coverage artifacts"

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

gpu:
	pip uninstall -y onnxruntime || true
	pip install -r requirements-gpu.txt

test:
	pytest

cov: test
	@echo "Open htmlcov/index.html"

run:
	uvicorn app.main:app --reload --port 8000

bench:
	python scripts/benchmark.py

charts:
	python scripts/generate_charts.py

clean:
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
