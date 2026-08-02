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
| Text to speech | Fallback connected, real model optional | Real OmniVoice is disabled by default for CPU deployment. |
| Voice cloning | Partial | Local profile/sample flow exists; production model hosting still needs planning. |
| Subtitle generation | Preview only | UI exists; real Whisper/subtitle backend still needs integration. |
| Shorts/video cuts | Partial | Upload/analyze/render pieces exist; advanced AI ranking/captions are not fully connected. |
| Supabase output sync | Optional | Requires Supabase env and database setup outside this deployment package. |
