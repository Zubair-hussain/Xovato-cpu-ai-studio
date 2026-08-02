"""Transcription and viral-moment ranking for the shorts pipeline.

Ported from AutoShorts' ``transcription.rs`` and ``llm.rs``, but provider
agnostic so the same code path runs locally with no API keys and on a hosted
site with them:

* **Transcription** - local ``faster-whisper`` (default, no key, nothing leaves
  the machine) or Deepgram.
* **Ranking** - a local heuristic ranker (default, no key) or any
  OpenAI-compatible chat API (DeepSeek, OpenAI, OpenRouter, Groq) or Anthropic.

Providers are chosen with environment variables, so the deployed instance can
use APIs for speed while a local run stays fully offline.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


class ShortsPipelineError(RuntimeError):
    """Raised when transcription or ranking fails."""


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    engine: str
    language: str
    segments: list[Segment] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments).strip()

    def as_dict(self) -> dict[str, object]:
        return {
            "engine": self.engine,
            "language": self.language,
            "segments": [asdict(segment) for segment in self.segments],
            "word_count": len(self.words),
        }


@dataclass
class Candidate:
    start_sec: float
    end_sec: float
    score: float
    hook: str
    rationale: str
    rank: int = 0

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    def as_dict(self) -> dict[str, object]:
        return {
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.end_sec, 2),
            "duration_sec": round(self.duration, 2),
            "score": round(self.score, 3),
            "hook": self.hook,
            "rationale": self.rationale,
            "rank": self.rank,
        }


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #

_whisper_lock = threading.Lock()
_whisper_model = None
_whisper_size: str | None = None


def _local_whisper(audio_path: Path, model_size: str) -> Transcript:
    global _whisper_model, _whisper_size

    try:
        from faster_whisper import WhisperModel
    except ImportError as error:  # pragma: no cover - dependency guard
        raise ShortsPipelineError(
            "faster-whisper is not installed. Install it, or set SHORTS_TRANSCRIBE_PROVIDER=deepgram."
        ) from error

    with _whisper_lock:
        if _whisper_model is None or _whisper_size != model_size:
            # int8 is what makes this usable on a CPU-only box.
            _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            _whisper_size = model_size
        model = _whisper_model

    try:
        raw_segments, info = model.transcribe(str(audio_path), beam_size=1, word_timestamps=True)
        segments: list[Segment] = []
        words: list[Word] = []
        for segment in raw_segments:
            text = (segment.text or "").strip()
            if text:
                segments.append(Segment(float(segment.start), float(segment.end), text))
            for word in segment.words or []:
                token = (word.word or "").strip()
                if token:
                    words.append(Word(float(word.start), float(word.end), token))
    except Exception as error:
        raise ShortsPipelineError(f"Local transcription failed: {error}") from error

    if not segments:
        raise ShortsPipelineError("No speech was found in this media.")

    return Transcript(engine=f"faster-whisper-{model_size}", language=info.language or "en", segments=segments, words=words)


def _deepgram(audio_path: Path, api_key: str) -> Transcript:
    import httpx

    try:
        with audio_path.open("rb") as handle:
            response = httpx.post(
                "https://api.deepgram.com/v1/listen",
                params={"model": "nova-2", "smart_format": "true", "punctuate": "true"},
                headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"},
                content=handle.read(),
                timeout=600.0,
            )
    except Exception as error:
        raise ShortsPipelineError(f"Could not reach Deepgram: {error}") from error

    if response.status_code != 200:
        raise ShortsPipelineError(f"Deepgram returned HTTP {response.status_code}: {response.text[:300]}")

    payload = response.json()
    try:
        alternative = payload["results"]["channels"][0]["alternatives"][0]
    except (KeyError, IndexError) as error:
        raise ShortsPipelineError("Deepgram response had an unexpected shape.") from error

    words = [
        Word(float(w["start"]), float(w["end"]), str(w.get("punctuated_word") or w.get("word", "")))
        for w in alternative.get("words", [])
    ]

    segments: list[Segment] = []
    for paragraph in alternative.get("paragraphs", {}).get("paragraphs", []):
        for sentence in paragraph.get("sentences", []):
            segments.append(Segment(float(sentence["start"]), float(sentence["end"]), str(sentence["text"])))

    if not segments and words:
        segments = _group_words_into_segments(words)
    if not segments:
        raise ShortsPipelineError("Deepgram found no speech in this media.")

    return Transcript(engine="deepgram-nova-2", language="en", segments=segments, words=words)


def _group_words_into_segments(words: list[Word], max_gap: float = 0.6) -> list[Segment]:
    segments: list[Segment] = []
    bucket: list[Word] = []

    for word in words:
        if bucket and word.start - bucket[-1].end > max_gap:
            segments.append(Segment(bucket[0].start, bucket[-1].end, " ".join(w.text for w in bucket)))
            bucket = []
        bucket.append(word)

    if bucket:
        segments.append(Segment(bucket[0].start, bucket[-1].end, " ".join(w.text for w in bucket)))
    return segments


def transcribe_audio(audio_path: Path, *, provider: str | None = None, model_size: str | None = None) -> Transcript:
    chosen = (provider or os.getenv("SHORTS_TRANSCRIBE_PROVIDER") or "local").strip().lower()

    if chosen == "deepgram":
        key = os.getenv("DEEPGRAM_API_KEY", "").strip()
        if not key:
            raise ShortsPipelineError("DEEPGRAM_API_KEY is not set.")
        return _deepgram(audio_path, key)

    return _local_whisper(audio_path, model_size or os.getenv("SHORTS_WHISPER_MODEL", "base"))


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #

# Openings that tend to make someone keep watching.
_HOOK_PATTERNS = (
    (re.compile(r"^(what|why|how|who|when|where|which)\b", re.I), 0.30),
    (re.compile(r"\?$"), 0.18),
    (re.compile(r"^(the|a|an)?\s*(secret|truth|reason|problem|mistake|trick|lesson)\b", re.I), 0.26),
    (re.compile(r"\b(never|always|nobody|everyone|no one|worst|best|biggest)\b", re.I), 0.20),
    (re.compile(r"\b(you|your)\b", re.I), 0.12),
    (re.compile(r"^\s*\d+\b|\b\d+%|\b\d+x\b", re.I), 0.16),
    (re.compile(r"\b(here'?s|listen|imagine|look|stop|don'?t)\b", re.I), 0.16),
    (re.compile(r"\b(because|so that|which means|that'?s why)\b", re.I), 0.10),
)

_FILLER = re.compile(r"\b(um|uh|erm|like|you know|i mean|sort of|kind of|basically)\b", re.I)


def _sentences(transcript: Transcript) -> list[Segment]:
    """Split segments on terminal punctuation so windows start at real boundaries."""
    result: list[Segment] = []
    for segment in transcript.segments:
        pieces = re.split(r"(?<=[.!?])\s+", segment.text.strip())
        pieces = [piece for piece in pieces if piece]
        if len(pieces) <= 1:
            result.append(segment)
            continue

        total = max(len(segment.text), 1)
        cursor = segment.start
        span = segment.end - segment.start
        for piece in pieces:
            share = span * (len(piece) / total)
            result.append(Segment(cursor, min(segment.end, cursor + share), piece))
            cursor += share
    return result


def rank_heuristic(
    transcript: Transcript,
    *,
    target_count: int,
    min_len: float,
    max_len: float,
) -> list[Candidate]:
    """Score self-contained, energetic, hook-led windows. No API key required.

    This finds well-formed segments with strong openings; it cannot judge whether
    an idea is genuinely interesting the way an LLM can, which is the honest
    limitation of running without one.
    """
    sentences = _sentences(transcript)
    if not sentences:
        return []

    ideal = (min_len + max_len) / 2
    candidates: list[Candidate] = []

    for index, first in enumerate(sentences):
        # Hook strength depends only on the opening line, so compute it once.
        hook_score = 0.0
        for pattern, weight in _HOOK_PATTERNS:
            if pattern.search(first.text.strip()):
                hook_score += weight
        hook_score = min(hook_score, 1.0)

        window: list[Segment] = [first]
        best: Candidate | None = None

        # Try every valid end point and keep the best-scoring one. Always
        # extending to max_len would swallow the rambling tail that usually
        # follows a good moment, so the length has to be chosen, not assumed.
        for follower in sentences[index:]:
            if follower is not first:
                if follower.end - first.start > max_len:
                    break
                window.append(follower)

            end = window[-1].end
            duration = end - first.start
            if duration < min_len:
                continue

            text = " ".join(item.text for item in window).strip()
            word_count = len(text.split())
            if word_count < 8:
                continue

            # Delivery pace - very slow or very fast both read as weak.
            pace = word_count / max(duration, 1e-6)
            pace_score = max(0.0, 1.0 - abs(pace - 2.8) / 2.8)

            # Closing on a full stop signals a complete thought.
            closure = 1.0 if text.rstrip().endswith((".", "!", "?")) else 0.35

            # Duration fit against the ideal clip length.
            fit = max(0.0, 1.0 - abs(duration - ideal) / max(ideal, 1e-6))

            # Filler density drags the score down.
            filler = len(_FILLER.findall(text)) / max(word_count, 1)
            filler_penalty = max(0.0, 1.0 - filler * 6.0)

            score = (
                hook_score * 0.34
                + pace_score * 0.20
                + closure * 0.16
                + fit * 0.16
                + filler_penalty * 0.14
            )

            if best is not None and score <= best.score:
                continue

            reasons = []
            if hook_score > 0.25:
                reasons.append("strong opening line")
            if pace_score > 0.6:
                reasons.append("lively pacing")
            if closure > 0.9:
                reasons.append("complete thought")
            if filler < 0.02:
                reasons.append("little filler")

            best = Candidate(
                start_sec=first.start,
                end_sec=end,
                score=score,
                hook=first.text.strip()[:120],
                rationale=", ".join(reasons) or "balanced segment",
            )

        if best is not None:
            candidates.append(best)

    return _suppress_overlaps(candidates, target_count)


def _suppress_overlaps(candidates: list[Candidate], target_count: int) -> list[Candidate]:
    """Keep the best non-overlapping windows, best first."""
    ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
    kept: list[Candidate] = []

    for candidate in ordered:
        if len(kept) >= target_count:
            break
        overlaps = any(
            candidate.start_sec < existing.end_sec and existing.start_sec < candidate.end_sec
            for existing in kept
        )
        if not overlaps:
            kept.append(candidate)

    kept.sort(key=lambda item: item.score, reverse=True)
    for position, candidate in enumerate(kept, start=1):
        candidate.rank = position
    return kept


_LLM_PROMPT = """You find viral short-form moments in transcripts.

Return ONLY a JSON array. Each item:
{{"start_sec": number, "end_sec": number, "score": number 0-1, "hook": string, "rationale": string}}

Rules:
- Each moment must be {min_len:.0f}-{max_len:.0f} seconds long.
- Moments must not overlap.
- Start on a complete sentence and end on a complete thought.
- "hook" is the opening line, quoted from the transcript.
- "rationale" is one short sentence on why it would perform.
- Return at most {count} moments, best first.

Transcript with timestamps:
{transcript}
"""


def rank_with_llm(
    transcript: Transcript,
    *,
    target_count: int,
    min_len: float,
    max_len: float,
) -> list[Candidate]:
    """Rank via any OpenAI-compatible chat API, or Anthropic."""
    import httpx

    provider = (os.getenv("SHORTS_LLM_PROVIDER") or "deepseek").strip().lower()
    lines = "\n".join(f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in transcript.segments)
    # Keep the request bounded; very long transcripts get truncated.
    prompt = _LLM_PROMPT.format(
        min_len=min_len, max_len=max_len, count=target_count, transcript=lines[:60000]
    )

    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise ShortsPipelineError("ANTHROPIC_API_KEY is not set.")
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={
                "model": os.getenv("SHORTS_LLM_MODEL", "claude-sonnet-4-5"),
                "max_tokens": 4000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=180.0,
        )
        if response.status_code != 200:
            raise ShortsPipelineError(f"Anthropic returned HTTP {response.status_code}: {response.text[:300]}")
        content = "".join(block.get("text", "") for block in response.json().get("content", []))
    else:
        endpoints = {
            "deepseek": ("https://api.deepseek.com/v1/chat/completions", "DEEPSEEK_API_KEY", "deepseek-chat"),
            "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY", "gpt-4o-mini"),
            "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY", "openai/gpt-4o-mini"),
            "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
        }
        if provider not in endpoints:
            raise ShortsPipelineError(f"Unknown SHORTS_LLM_PROVIDER '{provider}'.")

        url, env_name, default_model = endpoints[provider]
        key = os.getenv(env_name, "").strip()
        if not key:
            raise ShortsPipelineError(f"{env_name} is not set.")

        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("SHORTS_LLM_MODEL", default_model),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
            },
            timeout=180.0,
        )
        if response.status_code != 200:
            raise ShortsPipelineError(f"{provider} returned HTTP {response.status_code}: {response.text[:300]}")
        content = response.json()["choices"][0]["message"]["content"]

    return _parse_llm_candidates(content, target_count)


def _parse_llm_candidates(content: str, target_count: int) -> list[Candidate]:
    """Pull the JSON array out of an LLM reply, tolerating code fences."""
    match = re.search(r"\[.*\]", content, re.S)
    if not match:
        raise ShortsPipelineError("The ranking model did not return JSON.")

    try:
        rows = json.loads(match.group(0))
    except ValueError as error:
        raise ShortsPipelineError("The ranking model returned malformed JSON.") from error

    candidates: list[Candidate] = []
    for row in rows:
        try:
            candidates.append(
                Candidate(
                    start_sec=float(row["start_sec"]),
                    end_sec=float(row["end_sec"]),
                    score=float(row.get("score", 0.5)),
                    hook=str(row.get("hook", ""))[:120],
                    rationale=str(row.get("rationale", ""))[:240],
                )
            )
        except (KeyError, TypeError, ValueError):
            continue  # skip malformed rows rather than failing the whole job

    if not candidates:
        raise ShortsPipelineError("The ranking model returned no usable moments.")

    return _suppress_overlaps(candidates, target_count)


def rank_moments(
    transcript: Transcript,
    *,
    target_count: int = 5,
    min_len: float = 15.0,
    max_len: float = 60.0,
    provider: str | None = None,
) -> tuple[list[Candidate], str]:
    """Rank moments, falling back to the heuristic if the LLM path is unusable."""
    chosen = (provider or os.getenv("SHORTS_RANKING_PROVIDER") or "heuristic").strip().lower()

    if chosen in {"llm", "deepseek", "openai", "anthropic", "openrouter", "groq"}:
        if chosen != "llm":
            os.environ.setdefault("SHORTS_LLM_PROVIDER", chosen)
        try:
            return rank_with_llm(transcript, target_count=target_count, min_len=min_len, max_len=max_len), "llm"
        except ShortsPipelineError as error:
            # A missing key or a flaky API should degrade, not fail the whole job.
            logger.warning(
                "LLM ranking unavailable; using the heuristic ranker",
                extra={"module_name": "shorts", "action": "llm-fallback", "warning": str(error)},
            )

    return rank_heuristic(transcript, target_count=target_count, min_len=min_len, max_len=max_len), "heuristic"
