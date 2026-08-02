# Models

`deployment/models` is the CPU-only model root.

## Why We Need It

The backend needs stable model paths in production. Keeping models under one folder makes deployment, mounting, and copying much easier.

## CPU-Only Layout

```text
models/
├── segmentation/
├── face/
├── omnivoice/
├── whisper-cpu/
└── audio-cpu/
```

## Model Purposes

| Folder | Feature | Why needed |
| --- | --- | --- |
| `segmentation/` | Background removal | Stores `u2netp.onnx` for CPU object/person segmentation. |
| `face/` | Face restoration | Stores `yunet.onnx` and `gpen_bfr_256.onnx` for CPU face detect/restore. |
| `omnivoice/` | Text to speech | Optional heavy CPU TTS model folder. Disabled by default for deploy safety. |
| `whisper-cpu/` | Subtitles/transcription | Future small/base CPU Whisper models for real captions. |
| `audio-cpu/` | Audio enhancement | Placeholder for optional CPU audio models; current enhancement is mostly code-based. |

## Copy Current Local Models

From project root:

```powershell
.\deployment\models\copy-models.ps1
```

