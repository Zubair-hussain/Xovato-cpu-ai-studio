# Feature Status

This app has a polished studio UI, but not every listed model is fully connected yet.

## Why We Need This

Deployment is easier when you know which workflows are real, partial, or preview-only. This avoids deploying a UI label and expecting a model that is not included.

## Current Status

| Feature | Status | Notes |
| --- | --- | --- |
| Image enhancement | Connected | CPU code-based image enhancement is available. |
| Background removal | Connected if model exists | Needs `models/segmentation/u2netp.onnx`. |
| Face restoration | Connected if models exist | Needs `models/face/yunet.onnx` and `models/face/gpen_bfr_256.onnx`. |
| Audio enhancement | Connected | CPU code-based enhancement is available. |
| Text to speech | Fallback connected, real model optional | Real OmniVoice is disabled by default for CPU deployment; `/v1/audio/speech` now returns real WAV. |
| Voice cloning | Connected (consent-gated) | Profile/sample flow works; `consent_verified` must be true to create a clone. |
| Voice design | Connected | Gender/age/accent/pitch map onto the model's real instruct attributes. |
| Subtitle generation | Connected via shorts pipeline | `/v1/audio/transcriptions` runs faster-whisper; the `/voice/transcriptions` job is flagged `preview`. |
| Shorts/video cuts | Connected | Upload/analyze/render work; ranking is heuristic by default, LLM when a key is set. |
| Supabase output sync | Optional | Requires the Supabase **service-role** key and database setup outside this package. |

Endpoints that are still stubs (`/voice/transcriptions`, `/voice/dubbing`,
`/voice/audiobooks`, `/voice/stories`) now return a job whose `result.preview`
is `true`, instead of reporting a completed file that does not exist.
