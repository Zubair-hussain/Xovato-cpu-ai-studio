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

## Model cards & accuracy

Every model here was chosen for the **best accuracy that still runs on a CPU**.
The figures below are the models' **published/reported** accuracy on their
standard benchmarks - they describe the model, not this integration, and real
results vary with input quality. Sources are linked.

| Model | Feature | Size | Reported accuracy / quality | Why this one |
| --- | --- | --- | --- | --- |
| **U²-Net-p** (`u2netp.onnx`) | Background removal | ~4.5 MB | maxFβ ≈ 0.911, MAE ≈ 0.041 on ECSSD (the small variant of U²-Net) | Best salient-object accuracy per MB on CPU; the guided-filter refinement recovers edges the 320² net misses. |
| **YuNet** (`yunet.onnx`) | Face detection | ~230 KB | ≈ 0.834 mAP (easy) on WIDER FACE (libfacedetection/OpenCV Zoo) | Tiny, fast, 5-landmark output that drives the FFHQ alignment for restoration. |
| **GPEN-BFR-256** (`gpen_bfr_256.onnx`) | Face restoration | ~72 MB | Blind face restoration on par with GFPGAN; strong perceptual quality (LPIPS/FID) on CelebA-Test at 256² | Generative restoration that runs on CPU; 256² keeps it affordable. |
| **OmniVoice** (`k2-fsa/OmniVoice`) | Text to speech / cloning | ~2.5 GB | Zero-shot TTS with speaker-similarity and low WER in its model card; quality tracks step count & guidance | Attribute-controllable, clone-capable, permissive; int8 makes CPU usable (slow). |
| **faster-whisper** (`base`) | Transcription / subtitles | ~140 MB | Whisper-`base` ≈ 10-12% WER on LibriSpeech test-clean; `small`/`medium` lower it further | CTranslate2 int8 is the fastest accurate ASR on CPU; size is a speed/accuracy dial. |

**Accuracy dials you control**

- **Background removal:** raise `edge_softness` for hair/fur; the guided filter
  radius scales with image size automatically.
- **Face restoration:** `blend` (0-1) trades restored detail against preserving
  the original likeness.
- **TTS:** more `num_step` and higher `guidance_scale` = higher quality, slower.
  CPU deploy defaults to 8 steps / 1.6 guidance; the GPU path uses ~32 / 2.0.
- **Transcription:** pick `tiny`→`large-v3` to trade speed for word accuracy
  (`SHORTS_WHISPER_MODEL`).

Sources: U²-Net (Qin et al., *Pattern Recognition* 2020), YuNet
(libfacedetection / OpenCV Zoo), GPEN (Yang et al., CVPR 2021), OmniVoice and
faster-whisper model cards. Verify current numbers at the source before quoting
them - see the [Model Test notebook](./model_test.ipynb) to measure on your data.

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

## Download & test the models

Weights are gitignored (code ships, models do not). Get them and sanity-check
their accuracy with the notebook in this folder:

- **[model_test.ipynb](./model_test.ipynb)** - downloads each model, runs a real
  inference sanity check, and prints size/latency/accuracy signals. Works in
  Colab (GPU) or locally (CPU).

Direct sources (same as the Colab notebook uses):

| File | Source |
| --- | --- |
| `segmentation/u2netp.onnx` | `danielgatis/rembg` release assets |
| `face/yunet.onnx` | `opencv/opencv_zoo` (Git LFS media URL) |
| `face/gpen_bfr_256.onnx` | `facefusion/facefusion-assets` release assets |
| `omnivoice/k2-fsa_OmniVoice/` | `k2-fsa/OmniVoice` on Hugging Face (~2.5 GB) |

## What Is Not Included

- No RealESRGAN GPU `.pth` deployment slot
- No GFPGAN GPU `.pth` deployment slot
- No CUDA-only model requirement
- No GPU Docker base image

