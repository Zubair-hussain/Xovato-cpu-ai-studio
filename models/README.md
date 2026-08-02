# CPU-Only Model Folder

This deployment package is CPU-friendly by default. It does not require CUDA, GPU drivers, or GPU-only `.pth` model slots.

## Folder Layout

```text
deployment/models/
├── segmentation/
│   └── u2netp.onnx
├── face/
│   ├── yunet.onnx
│   └── gpen_bfr_256.onnx
├── omnivoice/
│   └── k2-fsa_OmniVoice/
├── whisper-cpu/
└── audio-cpu/
```

## CPU-Friendly Defaults

`ENHANCEAI_USE_REAL_OMNIVOICE=false`

This means deployment can start without loading the multi-GB OmniVoice model. The API can still use the lightweight WAV preview/fallback path.

## Current Local Model Sources

| Feature | Copy from | Copy to | CPU note |
| --- | --- | --- | --- |
| Background removal | `../backend/data/models/segmentation/u2netp.onnx` | `models/segmentation/u2netp.onnx` | Small CPU ONNX |
| Face detection | `../backend/data/models/face/yunet.onnx` | `models/face/yunet.onnx` | Small CPU ONNX |
| Face restoration | `../backend/data/models/face/gpen_bfr_256.onnx` | `models/face/gpen_bfr_256.onnx` | CPU-capable, moderate |
| Text to speech | `../backend/data/models/k2-fsa_OmniVoice` | `models/omnivoice/k2-fsa_OmniVoice` | CPU-capable but very heavy |
| Subtitles/transcription | not added yet | `models/whisper-cpu` | Use small/base CPU Whisper later |

## Copy CPU Models

From the project root:

```powershell
.\deployment\models\copy-models.ps1
```

The script copies the CPU-compatible local models into the deployment model folders.

## What Is Not Included

- No RealESRGAN GPU `.pth` deployment slot
- No GFPGAN GPU `.pth` deployment slot
- No CUDA-only model requirement
- No GPU Docker base image

