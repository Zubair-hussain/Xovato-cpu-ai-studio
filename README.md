# Xovato CPU-Only Backend Deployment

This folder is a clean CPU-only backend package for deploying the API on your non-Docker host. It does not include Docker files or Supabase SQL setup.

## Table Of Contents

| Section | Description |
| --- | --- |
| [Folder Map](#folder-map) | What each file/folder is for. |
| [Deployment Steps](#deployment-steps) | Recommended deploy order. |
| [Docs](#docs) | Detailed docs for app, models, environment, and feature status. |
| [CPU Model Policy](#cpu-model-policy) | Why this package avoids GPU-only model assumptions. |
| [Tests](#tests) | Deployment package test command. |
| [Quick Commands](#quick-commands) | Startup and local check commands. |

## Folder Map

| Path | Why we need it |
| --- | --- |
| `app/` | FastAPI backend code prepared for deployment. |
| `app/model_manager.py` | Lists CPU model paths and reports whether model files exist. |
| `app/services/` | Service facade folder matching the clean deployment structure. |
| `app/utils/` | Small file/cleanup helpers for deployment work. |
| `models/` | CPU-only model root for segmentation, face, optional OmniVoice, Whisper CPU, and audio CPU assets. |
| `docs/` | Detailed explanation files for each deployment area. |
| `tests/` | Deployment package tests. |
| `requirements.txt` | Python dependencies for the deployed backend. |
| `frontend.env.example` | Frontend environment variable template. |
| `backend.env.example` | Backend environment variable template with CPU-safe defaults. |
| `start-backend.txt` | Backend start command for hosts that ask for it separately. |

## Deployment Steps

| Step | Action | Why |
| --- | --- | --- |
| 1 | Configure backend env from `backend.env.example`. | Gives FastAPI CPU runtime and optional Supabase output settings. |
| 2 | Copy CPU model files with `models/copy-models.ps1` if needed. | Adds optional local models for CPU inference. |
| 3 | Run deployment tests. | Confirms the package has the expected files and CPU defaults. |
| 4 | Deploy this `deployment` folder as the backend API. | Runs the FastAPI API. |
| 5 | Deploy `../xovato-app` as the frontend. | Runs the user-facing Next.js app. |
| 6 | Configure frontend env from `frontend.env.example`. | Lets the Next.js app reach backend and optional Supabase. |

## Docs

| Doc | What it explains |
| --- | --- |
| [Docs Index](./docs/README.md) | All deployment docs in one place. |
| [App Package](./docs/app-package.md) | Why `app/` exists and how the backend is packaged. |
| [Models](./docs/models.md) | CPU model folders and why each model is needed. |
| [Environment](./docs/environment.md) | Frontend/backend env variables and secrets. |
| [Feature Status](./docs/feature-status.md) | Which features are connected, partial, or preview-only. |

## CPU Model Policy

This deployment folder is intentionally CPU-only:

- no CUDA requirement
- no GPU Docker image
- no RealESRGAN/GFPGAN GPU `.pth` deployment requirement
- real OmniVoice is optional and disabled by default with `ENHANCEAI_USE_REAL_OMNIVOICE=false`
- background removal and face restoration use CPU ONNX paths when model files are present

## Tests

From the `deployment` folder:

```powershell
..\backend\.venv\Scripts\python -m pytest tests
```

## Quick Commands

Install backend dependencies on your host:

```bash
pip install -r requirements.txt
```

Copy CPU-compatible models:

```powershell
.\models\copy-models.ps1
```

Run backend from this deployment folder:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

