# Documentation

Everything about the CPU-only Xovato backend lives here. Start with the root
[`../README.md`](../README.md) for the overview and feature list, then dive in.

## Guides

| Doc | Purpose |
| --- | --- |
| [App Package](./app-package.md) | How the FastAPI backend is packaged for deployment. |
| [Models](./models.md) | The CPU model folders, which features need them, and per-model accuracy. |
| [Environment](./environment.md) | Frontend/backend env variables and which ones are secret. |
| [Feature Status](./feature-status.md) | Which tools are connected, preview, or need a model. |
| [API Samples](./api-samples.md) | A runnable curl request for every feature endpoint. |
| [Samples](./samples.md) | Real before/after demo output for each feature. |
| [Benchmarks](./benchmarks.md) | Measured CPU numbers and an honest comparison to hosted services. |
| [Colab](./colab.md) | The GPU Colab notebook that runs every feature from one picker. |
| [Per-model Colab](../colab/README.md) | One notebook per model: test, benchmark, accuracy, training notes. |
| [GPU](./gpu.md) | Optional NVIDIA GPU acceleration (torch, onnxruntime-gpu, pycuda, cuda-python). |
| [Licensing](./licensing.md) | Optional offline key-based model licensing (root + admin chain). |

## Project & community

| Doc | Purpose |
| --- | --- |
| [Changelog](./CHANGELOG.md) | What changed, grouped by theme. |
| [Contributing](./CONTRIBUTING.md) | Setup, tests, conventions. |
| [Issues log](./ISSUES.md) | The 38 audited issues and their resolutions. |
| [Security policy](./SECURITY.md) | How to report vulnerabilities; built-in defenses. |
| [Code of conduct](./CODE_OF_CONDUCT.md) | Community expectations. |

## Assets

- `assets/` - benchmark charts, sample media, and example logs (`sample-logs.jsonl`).
