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

## Accuracy

Each model is the most accurate option that still runs on a CPU. Full model
cards with reported benchmark numbers, sources, and the accuracy dials you
control (edge softness, blend, TTS steps, whisper size) are in
[`../models/README.md`](../models/README.md#model-cards--accuracy). Measure them
on your own data with [`../models/model_test.ipynb`](../models/model_test.ipynb).

| Model | Feature | Reported quality (see model card for source) |
| --- | --- | --- |
| U²-Net-p | Background removal | maxFβ ≈ 0.91, MAE ≈ 0.04 (ECSSD) |
| YuNet | Face detection | ≈ 0.83 mAP easy (WIDER FACE) |
| GPEN-BFR-256 | Face restoration | Blind restoration on par with GFPGAN |
| OmniVoice | TTS / cloning | Zero-shot, quality scales with steps/guidance |
| faster-whisper base | Transcription | ≈ 10-12% WER (LibriSpeech test-clean) |

## Copy Current Local Models

From project root:

```powershell
.\deployment\models\copy-models.ps1
```

