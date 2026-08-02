# App Package

`deployment/app` contains the FastAPI backend code prepared for deployment.

## Why We Need It

The working backend lives in `../backend/app`, but deployment platforms usually expect a compact package with:

- application code
- requirements
- model folder
- start command

This folder gives you that clean package without changing your working development backend.

## Important Files

| Path | Why it exists |
| --- | --- |
| `app/main.py` | FastAPI entrypoint and API routes. |
| `app/config.py` | CPU-friendly deployment settings and model paths. |
| `app/model_manager.py` | Reports which CPU model files are present. |
| `app/services/` | Service facade folder matching the deploy structure you requested. |
| `app/utils/` | Small deployment utilities for files and cleanup. |

## Start Command

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```
