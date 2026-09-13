# Google Colab (GPU)

`Xovato_AI_Studio_Colab.ipynb` at the repo root runs every model feature in one notebook on a
Colab GPU, with a dropdown that switches between features.

## Why We Need It

This package is deliberately CPU-only so it can deploy to a plain host. That makes the heavy
features slow or optional: real OmniVoice is disabled by default, and Whisper runs int8 on CPU.
The notebook is the opposite end of the same code — identical backend functions, GPU underneath —
so features can be tried, demoed, and quality-checked without changing the deployment package.

## Open It

| Step | Action |
| --- | --- |
| 1 | Upload the notebook to [colab.research.google.com](https://colab.research.google.com), or open it from GitHub. |
| 2 | `Runtime` → `Change runtime type` → `T4 GPU` → Save. |
| 3 | Run cells ① → ⑤ in order, once per session. |
| 4 | Use the Feature Studio in cell ⑤. |

## Cells

| Cell | What it does |
| --- | --- |
| ① | Reports GPU, CPU, RAM, disk and ffmpeg. |
| ② | Clones this repo into `/content` and installs `onnxruntime-gpu`, `faster-whisper`, `omnivoice`. |
| ③ | Downloads `u2netp.onnx`, `yunet.onnx`, `gpen_bfr_256.onnx` and the `k2-fsa/OmniVoice` checkpoint. |
| ④ | Applies the GPU patches described below. |
| ⑤ | Feature Studio: the feature dropdown, its options, and the results. |
| ⑥ | Optional — serves `app.main:app` on a private Colab proxy URL. |
| ⑦ | Optional — zips the outputs and frees VRAM. |

## How The GPU Patches Work

No file in `app/` is modified. Cell ④ replaces the device layer underneath the unchanged code:

| Target | CPU code in `app/` | Patch |
| --- | --- | --- |
| `onnxruntime` | `providers=["CPUExecutionProvider"]` in `image_processing.py` and `face_restoration.py` | `ort.InferenceSession` is replaced with a factory that prefers `CUDAExecutionProvider` and falls back to CPU. Both modules import onnxruntime inside their session builders, so the swap reaches them. |
| OmniVoice | `voice_engine.load()` loads float32 on CPU and applies int8 dynamic quantisation | `VoiceEngine.load` is replaced with a float16 CUDA load. int8 quantisation is a CPU-only win and is skipped. |
| faster-whisper | `WhisperModel(device="cpu", compute_type="int8")` in `shorts_pipeline.py` | `faster_whisper.WhisperModel` is subclassed to force `cuda` / `float16`, falling back to CPU int8 if the CUDA libraries are missing. |
| ffmpeg | `-c:v libx264` in `shorts_media.render_clip` | `shorts_media._run` rewrites the command to `h264_nvenc` with `-rc vbr -cq`, and retries the original x264 command if NVENC refuses the stream. |

Because the patches live in the kernel, cell ⑥ starts uvicorn **in-process**. A subprocess server
would run unpatched, on CPU.

## The `app.services` Shim

The repo contains **both** `app/services.py` (the real `JobStore`, `VoiceStore`, circuit breakers
and health monitor) and an `app/services/` facade package. Python resolves a package before a
module of the same name, so this fails:

```
ImportError: cannot import name 'job_store' from 'app.services'
    (/content/Xovato-cpu-ai-studio/app/services/__init__.py)
```

That is not a Colab problem. `app/main.py:55` and `app/shorts_service.py:23` both do
`from app.services import job_store`, so the backend hits it on any machine where the package
directory exists — and `app/services/voice_cloning.py` imports `voice_store` from a package that
does not define it.

Cell ② writes `/content/xovato_bootstrap.py`, which loads `app/services.py` under the name
`app.services` and registers it in `sys.modules`. Every later import — including the ones inside
`app/` itself — then resolves to the real module, because `sys.modules` is consulted before the
filesystem. The shim is idempotent and no-ops if `app/services.py` is ever removed.

**This has now been fixed in the repo:** the real implementation lives in
`app/services/core.py` and is re-exported from `app/services/__init__.py`, so
`from app.services import job_store` resolves everywhere without any shim. The
notebook's `install()` still runs but simply reports *"not needed (layout
already fixed)"* because `app/services.py` no longer exists.

## Models The Notebook Downloads

| File | Source |
| --- | --- |
| `models/segmentation/u2netp.onnx` | `danielgatis/rembg` release assets |
| `models/face/yunet.onnx` | `opencv/opencv_zoo` (Git LFS media URL — the plain raw URL returns a pointer file) |
| `models/face/gpen_bfr_256.onnx` | `facefusion/facefusion-assets` release assets |
| `models/omnivoice/k2-fsa_OmniVoice/` | `k2-fsa/OmniVoice` on Hugging Face, ~2.5 GB |
| faster-whisper weights | downloaded by CTranslate2 on first transcription |

Every download is size-checked, so a Git LFS pointer or an error page is rejected instead of being
saved as a broken model.

## Limits

- Colab runtimes are ephemeral: models re-download on every fresh session (~3 GB with OmniVoice).
- The browser upload widget is slow above ~50 MB. Put large video in `/content` with the Files
  panel and paste the path instead.
- The job store is in-memory, so the shorts project list does not survive a runtime restart.
