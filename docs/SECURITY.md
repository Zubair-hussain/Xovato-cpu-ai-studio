# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Report privately via GitHub Security Advisories:
<https://github.com/Zubair-hussain/Xovato-cpu-ai-studio/security/advisories/new>

You can expect an acknowledgement within a few days. Please include:

- affected endpoint or module,
- a minimal reproduction,
- the impact (data exposure, DoS, RCE, etc.).

## Scope and hardening notes

This backend is designed to run behind your own auth/proxy. The following
defenses are built in and should not be weakened:

- **Uploads** are size-limited while streaming (image, audio, voice sample,
  shorts video) to prevent memory/disk exhaustion.
- **`GET /api/v1/files/{path}`** serves only the `outputs/` and `shorts/`
  subtrees; logs and the voice-profile index are not reachable.
- **Voice cloning** requires explicit `consent_verified: true`.
- **CORS** origins are restricted via `ENHANCEAI_CORS_ORIGINS`.
- **Supabase output sync** requires the service-role key and never reads a
  sibling frontend's `.env.local`.
- **Circuit breakers** pause segmentation/synthesis after repeated real failures.

The API itself ships **without authentication** - put it behind a gateway,
network policy, or reverse proxy that enforces access control before exposing it
to the public internet.

## Supported versions

The `main` branch receives security fixes.
