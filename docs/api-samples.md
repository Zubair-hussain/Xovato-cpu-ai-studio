# API Samples

One runnable sample per feature. Each maps to a selection in the Feature Studio
(the Colab notebook, cell ⑤) and to a module in the backend. Base URL assumes a
local server:

```bash
export BASE=http://localhost:8000
```

Start it with `uvicorn app.main:app --port 8000` (or `make run`).

| Feature (record selection) | Endpoint | Module |
| --- | --- | --- |
| Health / status | `GET /api/v1/health` | `system-health` |
| Text → Speech | `POST /api/v1/voice/speech` | `text-to-speech` |
| Stock voices | `GET /api/v1/voice/library` | `text-to-speech` |
| Voice styles | `GET /api/v1/voice/styles` | `text-to-speech` |
| Voice Cloning | `POST /api/v1/voice/profiles` | `voice-cloning` |
| Voice Design | `POST /api/v1/voice/design` | `voice-design` |
| Image Enhancement | `POST /api/v1/images/enhance` | `image-enhancement` |
| Background Removal | `POST /api/v1/images/background-removal` | `background-removal` |
| Face Restoration | `POST /api/v1/images/face-restoration` | `face-restoration` |
| Audio Enhancement | `POST /api/v1/audio/enhance` | `audio-enhancement` |
| Shorts (analyze) | `POST /api/v1/shorts/analyze` | `shorts-studio` |
| Shorts (render) | `POST /api/v1/shorts/{project_id}/render` | `shorts-studio` |
| OpenAI-compatible TTS | `POST /v1/audio/speech` | `text-to-speech` |

---

## Health

```bash
curl -s $BASE/api/v1/health | jq
```

## Text → Speech

Uses a stock voice (no reference audio, no cloning). In CPU deploy mode this
returns a preview WAV unless `ENHANCEAI_USE_REAL_OMNIVOICE=true`.

```bash
curl -s -X POST $BASE/api/v1/voice/speech \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from Xovato.","voice_id":"aria","emotion":"calm","response_format":"wav"}' | jq
# -> a Job with output_uri; download it:
#   curl -s "$BASE/api/v1/files/outputs/<job_id>.wav" -o speech.wav
```

List stock voices and styles:

```bash
curl -s $BASE/api/v1/voice/library | jq '.voices[].id'
curl -s $BASE/api/v1/voice/styles  | jq '.emotions[].id'
```

## Voice Cloning (consent required)

```bash
# 1. Upload a reference clip
curl -s -X POST $BASE/api/v1/voice/samples -F "sample=@reference.wav" | jq

# 2. Create the profile - consent_verified MUST be true
curl -s -X POST $BASE/api/v1/voice/profiles \
  -H "Content-Type: application/json" \
  -d '{"name":"My Voice","reference_uri":"local://voice_samples/<sample_id>.wav",
       "reference_text":"the exact words spoken in the clip","consent_verified":true}' | jq
```

## Voice Design

Attributes are mapped onto the model's real instruct vocabulary.

```bash
curl -s -X POST $BASE/api/v1/voice/design \
  -H "Content-Type: application/json" \
  -d '{"name":"Narrator","gender":"female","age":"young","accent":"british","pitch":1.1}' | jq
```

## Image Enhancement

```bash
curl -s -X POST $BASE/api/v1/images/enhance \
  -F "image=@photo.jpg" -F "output_format=jpg" -F "quality=92" \
  -D headers.txt -o enhanced.jpg
# The measurement report rides in a header:
grep -i x-image-analysis headers.txt | cut -d' ' -f2 | base64 -d | jq
```

## Background Removal

Needs `models/segmentation/u2netp.onnx` (503 until present).

```bash
curl -s -X POST $BASE/api/v1/images/background-removal \
  -F "image=@subject.jpg" -F "output_format=png" -o cutout.png
```

## Face Restoration

Needs `models/face/yunet.onnx` and `models/face/gpen_bfr_256.onnx`.

```bash
curl -s -X POST $BASE/api/v1/images/face-restoration \
  -F "image=@portrait.jpg" -F "blend=1.0" -o restored.jpg
```

## Audio Enhancement

```bash
curl -s -X POST $BASE/api/v1/audio/enhance \
  -F "audio=@recording.wav" -F "denoise_strength=0.6" -F "target_lufs=-16" \
  -o clean.wav
```

## Shorts

```bash
# 1. Analyze a long video (returns a job id; poll it)
curl -s -X POST $BASE/api/v1/shorts/analyze \
  -F "media=@talk.mp4" -F "target_count=5" -F "min_len=15" -F "max_len=60" | jq
# {"job_id":"...","project_id":"proj_...","status":"queued"}

# 2. Poll
curl -s $BASE/api/v1/shorts/jobs/<job_id> | jq '.progress, .result.candidates'

# 3. Render the chosen moments to 9:16
curl -s -X POST $BASE/api/v1/shorts/<project_id>/render \
  -H "Content-Type: application/json" \
  -d '{"clips":[{"start_sec":12.0,"end_sec":42.0,"hook":"the key point"}],"captions":true}' | jq
```

## OpenAI-compatible endpoints

Drop-in for clients that speak the OpenAI audio API (WAV only).

```bash
curl -s -X POST $BASE/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","voice":"aria","input":"Hello","response_format":"wav"}' -o openai.wav

curl -s -X POST $BASE/v1/audio/transcriptions -F "file=@clip.wav" | jq
```
