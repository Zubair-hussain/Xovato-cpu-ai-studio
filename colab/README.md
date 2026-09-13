# Per-Model Colab Notebooks

One notebook per model. Each is self-contained and runs on Colab (GPU) or
locally (CPU): setup -> data -> load -> inference test -> **benchmark** ->
**accuracy signal** -> training/fine-tuning notes -> copyright.

| Notebook | Model | Feature |
| --- | --- | --- |
| [01_u2netp_background_removal.ipynb](01_u2netp_background_removal.ipynb) | U²-Net-p | Background removal |
| [02_yunet_face_detection.ipynb](02_yunet_face_detection.ipynb) | YuNet | Face detection |
| [03_gpen_face_restoration.ipynb](03_gpen_face_restoration.ipynb) | GPEN-BFR-256 | Face restoration |
| [04_omnivoice_tts_cloning.ipynb](04_omnivoice_tts_cloning.ipynb) | OmniVoice | TTS & voice cloning |
| [05_whisper_transcription.ipynb](05_whisper_transcription.ipynb) | faster-whisper | Transcription / subtitles |

## Honest notes

- **Benchmarks** run on whatever machine executes the cell and print real timings.
- **Accuracy cells** compute what is feasible in a notebook (IoU on a test image,
  detection confidence, RTF) and cite each model's **published** benchmark number
  with its source - they do not claim to reproduce full paper benchmarks. Each
  notebook shows how to run the real dataset benchmark.
- **Training** is described honestly: these are pretrained models. Where full
  retraining needs a GPU cluster it is documented, not faked; where "training"
  really means zero-shot adaptation (OmniVoice cloning) that is shown live.
- Some notebooks need a **real input** (a face photo, a speech clip) - synthetic
  media has no faces or words, so those cells ask you to supply a file.

Reported accuracy for every model is summarised in
[../models/README.md](../models/README.md#model-cards--accuracy).

---
(c) Zubair Hussain - Xovato AI Studio. Keep this attribution in derived work.
