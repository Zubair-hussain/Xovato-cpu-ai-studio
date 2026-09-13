# GPU acceleration (optional)

The backend is **CPU-first by design** - it deploys to a plain host with no CUDA.
This page is for when you move to an NVIDIA GPU and want the heavy features
(real OmniVoice TTS, ONNX background removal / face restoration, faster-whisper)
to run on it.

> You do **not** need any of this for the default CPU deployment. Nothing in
> `requirements-gpu.txt` is imported unless the GPU code paths actually run.

## Requirements

| Component | Notes |
| --- | --- |
| NVIDIA GPU | Compute capability 6.1+ (GTX 10-series or newer) |
| NVIDIA driver | Recent enough for your CUDA version (`nvidia-smi` to check) |
| CUDA toolkit | 12.x recommended. `pycuda` needs `nvcc` at install time |
| Python | 3.11+ |

## Install

```bash
# 1. Base runtime (always)
pip install -r requirements.txt

# 2. The CPU and GPU ONNX Runtime wheels cannot coexist - remove the CPU one:
pip uninstall -y onnxruntime

# 3. GPU acceleration
pip install -r requirements-gpu.txt
```

`requirements-gpu.txt` pulls `torch` from the CUDA 12.1 wheel index. If your CUDA
is different, edit the `--extra-index-url` line (`cu118`, `cu124`, ...).

pip fetches the correct platform wheel for `torch`, `onnxruntime-gpu`, `pycuda`
and `cuda-python` from their indexes. **Do not vendor `.whl` files into the
repo** - they are hundreds of MB, platform- and CUDA-version-specific, and go
stale immediately.

## Turn the GPU paths on

```bash
export ENHANCEAI_USE_REAL_OMNIVOICE=true      # real TTS instead of preview
export ENHANCEAI_OMNIVOICE_QUANTIZE_INT8=false # int8 is a CPU trick; skip on GPU
```

`onnxruntime-gpu` selects the CUDA execution provider automatically when present,
so background removal and face restoration move to the GPU with no code change.

## Verify

```bash
python - <<'PY'
import torch, onnxruntime as ort
print("torch cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("onnxruntime providers:", ort.get_available_providers())
try:
    import pycuda.driver as drv; drv.init()
    print("pycuda devices:", drv.Device.count())
except Exception as e:
    print("pycuda:", e)
PY
```

You want to see `torch cuda: True`, `CUDAExecutionProvider` in the provider list,
and at least one pycuda device.

## Already have a GPU box for a quick demo?

The [Colab notebook](colab.md) does all of this for you: it installs the GPU
wheels and patches the device layer at runtime, running the unchanged `app/`
code on a free T4 GPU. Use it to try the GPU path without touching your deploy.
