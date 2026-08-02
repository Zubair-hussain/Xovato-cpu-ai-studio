# Environment

The deployment package has example env files:

- `../frontend.env.example`
- `../backend.env.example`

## Why We Need Them

The frontend needs public URLs. The backend needs CPU runtime settings and optional private Supabase credentials if you want output sync.

## Frontend Env

| Variable | Why needed |
| --- | --- |
| `NEXT_PUBLIC_BACKEND_URL` | Points the UI to your deployed API. |
| `NEXT_PUBLIC_BACKEND_PROXY_URL` | Keeps local/proxy API paths consistent. |
| `NEXT_PUBLIC_SUPABASE_URL` | Optional public Supabase project URL. |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Optional public anon key for client-safe Supabase calls. |

## Backend Env

| Variable | Why needed |
| --- | --- |
| `SUPABASE_URL` | Optional Supabase REST/storage endpoint base. |
| `SUPABASE_SERVICE_ROLE_KEY` | Optional private backend key for inserting outputs and uploading files. |
| `SUPABASE_OUTPUTS_BUCKET` | Optional bucket name for generated outputs. |
| `ENHANCEAI_ENVIRONMENT` | Marks runtime as production. |
| `ENHANCEAI_USE_REAL_OMNIVOICE` | Defaults false for CPU-friendly deploy. |
| `ENHANCEAI_OMNIVOICE_QUANTIZE_INT8` | Keeps optional CPU TTS lighter when enabled. |

Never expose `SUPABASE_SERVICE_ROLE_KEY` in frontend/browser env.

Supabase SQL is not bundled in this deployment folder. Keep database setup separate from backend deployment.
