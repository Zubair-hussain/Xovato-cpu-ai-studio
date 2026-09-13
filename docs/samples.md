# Feature Samples

Real output from the actual backend - every file here was produced by the same
functions the API calls, via [`scripts/generate_samples.py`](../scripts/generate_samples.py).
Nothing is mocked. Regenerate any time with:

```bash
python scripts/generate_samples.py
```

> The inputs are **synthetic test media** (generated, not real photos/voices) so
> the samples are reproducible and license-free. Quality on real content is
> typically better - drop in your own file to see.

## 🖼️ Image Enhancement

Analysis-driven correction: contrast recovery, colour revival, tone curve,
local contrast and blur-matched sharpening - chosen from the image's own
histogram.

![Image enhancement before/after](assets/samples/image_before_after.png)

| | File |
| --- | --- |
| Input (degraded) | [image_before.jpg](assets/samples/image_before.jpg) |
| Output (enhanced) | [image_after.jpg](assets/samples/image_after.jpg) |

*Measured 1280x853: exposure 0.57, contrast 0.05 → recovered black/white points,
lifted midtones, revived colour, added local contrast, sharpened to the measured
3.8px blur scale (7 corrections).*

## ✂️ Background Removal

u2netp ONNX segmentation refined against the image's own edges with a guided
filter, producing true per-pixel alpha (shown over a checkerboard).

![Background removal cutout](assets/samples/bg_cutout_on_checker.png)

| | File |
| --- | --- |
| Input (subject on clutter) | [bg_input.jpg](assets/samples/bg_input.jpg) |
| Output (transparent PNG) | [bg_cutout.png](assets/samples/bg_cutout.png) |
| Output on checkerboard | [bg_cutout_on_checker.png](assets/samples/bg_cutout_on_checker.png) |

*Isolated the subject from a busy 800x800 scene; foreground covers 15.6% of the
frame. Transparency is real alpha, not a crop.*

## 🔊 Audio Enhancement (1 minute)

A 60-second clip with hiss, 50 Hz mains hum and low rumble → spectral denoise →
BS.1770 loudness normalisation to −16 LUFS → true-peak limiter.

![Audio waveform before/after](assets/samples/audio_before_after.png)

| | File (60s WAV) |
| --- | --- |
| Input (noisy) | [audio_before.wav](assets/samples/audio_before.wav) |
| Output (cleaned) | [audio_after.wav](assets/samples/audio_after.wav) |

*60.0s, 44.1 kHz stereo. Loudness −14.5 → −16.0 LUFS, peak −4.0 → −4.8 dBFS,
rumble and hum removed below 80 Hz.*

## 🎙️ Text → Speech / 🧬 Voice Cloning

TTS output depends on the runtime, and these samples are the **CPU preview
engine** the backend ships with - on Windows that is the System.Speech (SAPI)
voice; elsewhere a tonal placeholder. They are honest samples of the default
path, **not** the neural OmniVoice model.

| | File |
| --- | --- |
| Text → Speech (preview) | [tts_preview.wav](assets/samples/tts_preview.wav) |
| Voice cloning workflow (preview) | [clone_preview.wav](assets/samples/clone_preview.wav) |

For the real neural voice, run OmniVoice on a GPU (or CPU with
`ENHANCEAI_USE_REAL_OMNIVOICE=true` and the checkpoint downloaded). Generate your
own with the [Model Test notebook](../models/model_test.ipynb),
the [per-model Colab](../colab/README.md), or:

```bash
curl -X POST http://localhost:8000/api/v1/voice/speech \
  -H "Content-Type: application/json" \
  -d '{"text":"This voice is generated locally.","voice_id":"aria","emotion":"calm"}'
```

> Why preview and not the neural model here? The OmniVoice checkpoint is ~2.5 GB
> and needs torch - too heavy to ship as a repo sample. The preview path is what
> a fresh CPU deploy actually returns, so it is the honest thing to show.

## 😀 Face Restoration

Needs a real face photo plus the YuNet + GPEN-BFR-256 models, so it is not part
of the synthetic sample set (YuNet correctly finds no face in generated shapes).
Run it on a real portrait:

```bash
curl -X POST http://localhost:8000/api/v1/images/face-restoration \
  -F "image=@portrait.jpg" -F "blend=1.0" -o restored.jpg
```

See [models.md](models.md) for where to get the model files and their accuracy.
