from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from supertokens_python.recipe.session import SessionContainer
from supertokens_python.recipe.session.framework.fastapi import verify_session

logger = logging.getLogger(__name__)

router = APIRouter()

_whisperx_model = None
_whisperx_lock: asyncio.Lock | None = None

CLIP_DURATION_SECONDS = 60


class ClipRequest(BaseModel):
    audio_url: str = Field(..., min_length=1)
    timestamp: float = Field(..., ge=0)
    pod_id: str = Field(..., min_length=1)
    episode_idx: int


class WordSegment(BaseModel):
    word: str
    start: float
    end: float


class ClipResponse(BaseModel):
    words: List[WordSegment]
    clip_start: float
    clip_end: float
    video_url: Optional[str] = None


def _get_whisperx_model():
    global _whisperx_model
    if _whisperx_model is not None:
        logger.debug("WhisperX model already cached, reusing")
        return _whisperx_model

    logger.info("Loading WhisperX model for the first time...")
    t0 = time.time()
    import warnings
    # Suppress PyTorch TF32 deprecation warning from pyannote/whisperx internals
    warnings.filterwarnings("ignore", message=".*TF32.*", category=UserWarning)
    # Suppress ONNX Runtime GPU discovery warning in Docker
    os.environ.setdefault("ORT_DISABLE_ALL_DEVICE_DISCOVERY", "1")
    import torch
    import torchaudio
    # Compatibility patches for pyannote-audio with torchaudio >= 2.1
    if not hasattr(torchaudio, 'AudioMetaData'):
        logger.info("Patching missing torchaudio.AudioMetaData")
        torchaudio.AudioMetaData = type('AudioMetaData', (), {})
    if not hasattr(torchaudio, 'list_audio_backends'):
        logger.info("Patching missing torchaudio.list_audio_backends")
        torchaudio.list_audio_backends = lambda: ['soundfile', 'sox']
    # PyTorch 2.6+ defaults weights_only=True for torch.load, but pyannote
    # checkpoints use OmegaConf classes. Patch torch.load to use weights_only=False.
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False  # Force False, override caller value
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
    logger.info("Patched torch.load to use weights_only=False for model loading")
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    logger.info("WhisperX device=%s, compute_type=%s", device, compute_type)
    model = whisperx.load_model("base", device, compute_type=compute_type, language="en")
    _whisperx_model = {"model": model, "device": device}
    logger.info("WhisperX model loaded in %.1fs", time.time() - t0)
    return _whisperx_model


def _resolve_audio_path(audio_url: str) -> str:
    pods_base = os.getenv("PODS_BASE", "")
    audio_url_prefix = os.getenv("AUDIO_URL_PREFIX", "")

    logger.debug("Resolving audio_url=%s (PODS_BASE=%s, AUDIO_URL_PREFIX=%s)",
                 audio_url, pods_base, audio_url_prefix)

    if audio_url_prefix and audio_url.startswith(audio_url_prefix):
        relative = audio_url[len(audio_url_prefix):].lstrip("/")
        local_path = Path(pods_base).expanduser() / relative
        logger.debug("Matched AUDIO_URL_PREFIX, local_path=%s, exists=%s", local_path, local_path.exists())
        if local_path.exists():
            return str(local_path)

    parsed = urlparse(audio_url)
    if parsed.path.startswith("/pods/"):
        relative = parsed.path[len("/pods/"):]
        local_path = Path(pods_base).expanduser() / relative
        logger.debug("Matched /pods/ prefix, local_path=%s, exists=%s", local_path, local_path.exists())
        if local_path.exists():
            return str(local_path)

    logger.debug("No local match, using audio_url as-is (remote): %s", audio_url)
    return audio_url


def _extract_clip(audio_source: str, timestamp: float, duration: int = CLIP_DURATION_SECONDS) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    command = [
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-t", str(duration),
        "-i", audio_source,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        tmp_path,
    ]

    logger.info("Running ffmpeg: %s", " ".join(command))
    t0 = time.time()

    try:
        proc = subprocess.run(command, check=True, capture_output=True, timeout=60)
        elapsed = time.time() - t0
        tmp_size = Path(tmp_path).stat().st_size if Path(tmp_path).exists() else 0
        logger.info("ffmpeg completed in %.1fs, output size=%d bytes", elapsed, tmp_size)
        if proc.stderr:
            logger.debug("ffmpeg stderr: %s", proc.stderr.decode(errors="ignore")[-500:])
    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - t0
        Path(tmp_path).unlink(missing_ok=True)
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
        logger.error("ffmpeg failed after %.1fs (exit %d): %s", elapsed, exc.returncode, stderr[-500:])
        raise RuntimeError(f"ffmpeg clip extraction failed: {stderr.strip()}")
    except subprocess.TimeoutExpired:
        Path(tmp_path).unlink(missing_ok=True)
        logger.error("ffmpeg timed out after 60s for source=%s", audio_source)
        raise RuntimeError("ffmpeg clip extraction timed out")

    return tmp_path


def _transcribe_clip(clip_path: str, timestamp_offset: float) -> List[Dict[str, Any]]:
    import whisperx

    logger.info("Starting transcription of %s (offset=%.1fs)", clip_path, timestamp_offset)
    t0 = time.time()

    cached = _get_whisperx_model()
    model = cached["model"]
    device = cached["device"]

    logger.debug("Loading audio from %s", clip_path)
    audio = whisperx.load_audio(clip_path)
    logger.debug("Audio loaded, length=%.1fs", len(audio) / 16000)

    logger.info("Running whisperx transcribe...")
    t1 = time.time()
    result = model.transcribe(audio, batch_size=16)
    logger.info("Transcription done in %.1fs, language=%s, segments=%d",
                time.time() - t1, result.get("language"), len(result.get("segments", [])))

    logger.info("Loading alignment model for language=%s", result.get("language"))
    t2 = time.time()
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device,
    )
    logger.info("Alignment model loaded in %.1fs", time.time() - t2)

    logger.info("Running alignment...")
    t3 = time.time()
    aligned = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    logger.info("Alignment done in %.1fs", time.time() - t3)

    words = []
    for segment in aligned.get("word_segments", []):
        words.append({
            "word": segment.get("word", ""),
            "start": round((segment.get("start", 0) or 0) + timestamp_offset, 3),
            "end": round((segment.get("end", 0) or 0) + timestamp_offset, 3),
        })

    logger.info("Transcription complete: %d words in %.1fs total", len(words), time.time() - t0)
    return words


# ============================================================================
# Video Generation Functions
# ============================================================================

VIDEO_WIDTH = 405
VIDEO_HEIGHT = 720
VIDEO_FPS = 24
MAX_CHARS_PER_LINE = 30
FONT_SIZE = 24
LINE_SPACING = 1.5  # Line height multiplier
LINE_HEIGHT_PX = int(FONT_SIZE * LINE_SPACING)  # 36px per line
VERTICAL_PADDING_RATIO = 0.10  # 10% padding on top and bottom

# Calculate visible lines based on usable vertical space
_usable_height = int(VIDEO_HEIGHT * (1 - 2 * VERTICAL_PADDING_RATIO))  # 80% of height
VISIBLE_LINES = _usable_height // LINE_HEIGHT_PX  # How many lines fit


def _format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format (H:MM:SS.cc)."""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _group_words_into_lines(
    words: List[Dict[str, Any]],
    max_chars: int = MAX_CHARS_PER_LINE
) -> List[List[Dict[str, Any]]]:
    """Group words into lines that fit the video width."""
    lines = []
    current_line: List[Dict[str, Any]] = []
    current_length = 0

    for word in words:
        word_text = word.get("word", "")
        word_len = len(word_text) + 1  # +1 for space

        if current_length + word_len > max_chars and current_line:
            lines.append(current_line)
            current_line = []
            current_length = 0

        current_line.append(word)
        current_length += word_len

    if current_line:
        lines.append(current_line)

    return lines


def _generate_ass_subtitles(
    words: List[Dict[str, Any]],
    clip_start: float,
    clip_duration: float = CLIP_DURATION_SECONDS
) -> str:
    """Generate ASS subtitle content with karaoke timing and scrolling.

    Scrolling behavior:
    - Initially, VISIBLE_LINES (12) are displayed centered on screen from time 0
    - Scrolling starts when playback reaches 50% of VISIBLE_LINES (line 6)
    - Text scrolls upward to reveal remaining content
    - Karaoke highlighting is controlled by \\kf tags, not line start/end times
    """
    lines = _group_words_into_lines(words)
    total_lines = len(lines)

    # Calculate if scrolling is needed
    needs_scroll = total_lines > VISIBLE_LINES
    visible_height = VISIBLE_LINES * LINE_HEIGHT_PX

    # Top padding (10% of height)
    top_padding = int(VIDEO_HEIGHT * VERTICAL_PADDING_RATIO)
    # First line starts at top padding
    first_line_y = top_padding

    # ASS header with styling
    header = f"""[Script Info]
Title: Clip Transcript
ScriptType: v4.00+
PlayResX: {VIDEO_WIDTH}
PlayResY: {VIDEO_HEIGHT}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{FONT_SIZE},&H0080FFFF,&H00A0A0A0,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,8,10,10,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []

    if needs_scroll:
        # Calculate scroll parameters
        # Scroll starts when we reach 50% of visible lines (line index VISIBLE_LINES // 2)
        scroll_trigger_line = VISIBLE_LINES // 2  # Line 6 for 12 visible lines
        extra_lines = total_lines - VISIBLE_LINES
        scroll_distance = extra_lines * LINE_HEIGHT_PX

        # Find the timestamp when scroll should start (when we reach the trigger line)
        scroll_start_time = 0.0
        if scroll_trigger_line < len(lines) and lines[scroll_trigger_line]:
            first_word = lines[scroll_trigger_line][0]
            scroll_start_time = first_word.get("start", clip_start) - clip_start

        # Scroll duration: from trigger to end of clip
        scroll_end_time = clip_duration
        scroll_duration_ms = int((scroll_end_time - scroll_start_time) * 1000)
        scroll_start_ms = int(scroll_start_time * 1000)

        for line_idx, line_words in enumerate(lines):
            if not line_words:
                continue

            # Get the time when this line starts speaking (for karaoke delay)
            line_speak_start = line_words[0].get("start", clip_start) - clip_start

            # Build karaoke text with timing
            # Add initial delay so highlighting starts when the line is spoken
            karaoke_text = ""
            if line_speak_start > 0:
                # Add delay in centiseconds before first word highlights
                delay_cs = int(line_speak_start * 100)
                karaoke_text = f"{{\\k{delay_cs}}}"

            for w in line_words:
                word_duration = int((w.get("end", 0) - w.get("start", 0)) * 100)
                word_duration = max(10, word_duration)  # Minimum 0.1 second
                karaoke_text += f"{{\\kf{word_duration}}}{w.get('word', '')} "

            # Initial position: all lines start centered
            line_y_start = first_line_y + (line_idx * LINE_HEIGHT_PX)

            # End position after scroll
            line_y_end = line_y_start - scroll_distance

            # ALL lines are visible from time 0 until end of clip
            # The karaoke \k delay handles when highlighting starts
            start_ts = _format_ass_time(0)
            end_ts = _format_ass_time(clip_duration)

            # Use move for scrolling effect
            # Move starts at scroll_start_ms and ends at scroll_start_ms + scroll_duration_ms
            move_tag = f"{{\\move({VIDEO_WIDTH // 2},{line_y_start},{VIDEO_WIDTH // 2},{line_y_end},{scroll_start_ms},{scroll_start_ms + scroll_duration_ms})}}"
            events.append(
                f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{move_tag}{karaoke_text.strip()}"
            )
    else:
        # No scrolling needed - center text vertically, all visible from start
        for line_idx, line_words in enumerate(lines):
            if not line_words:
                continue

            # Get the time when this line starts speaking (for karaoke delay)
            line_speak_start = line_words[0].get("start", clip_start) - clip_start

            # Build karaoke text with timing
            karaoke_text = ""
            if line_speak_start > 0:
                # Add delay in centiseconds before first word highlights
                delay_cs = int(line_speak_start * 100)
                karaoke_text = f"{{\\k{delay_cs}}}"

            for w in line_words:
                word_duration = int((w.get("end", 0) - w.get("start", 0)) * 100)
                word_duration = max(10, word_duration)  # Minimum 0.1 second
                karaoke_text += f"{{\\kf{word_duration}}}{w.get('word', '')} "

            line_y = first_line_y + (line_idx * LINE_HEIGHT_PX)

            # All lines visible from time 0
            start_ts = _format_ass_time(0)
            end_ts = _format_ass_time(clip_duration)

            pos_tag = f"{{\\pos({VIDEO_WIDTH // 2},{line_y})}}"
            events.append(
                f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{pos_tag}{karaoke_text.strip()}"
            )

    # Debug: print events array
    if os.getenv("DEBUG_ASS_EVENTS"):
        logger.info("=== ASS Events Debug (%d lines, %d events) ===", total_lines, len(events))
        for i, event in enumerate(events):
            logger.info("Event %d: %s", i, event[:200] + "..." if len(event) > 200 else event)
        logger.info("=== End ASS Events ===")

    return header + "\n".join(events)


def _find_cover_image(book_folder: Path) -> Optional[Path]:
    """Find cover image in book folder."""
    for ext in ["jpg", "jpeg", "png", "webp"]:
        cover_path = book_folder / f"cover.{ext}"
        if cover_path.exists():
            return cover_path
    return None


def _get_book_folder(pod_id: str, user_id: str, pgdb: Any) -> Optional[Path]:
    """Get book folder path from pod_id using database lookup."""
    pods_base = Path(os.getenv("PODS_BASE", "")).expanduser()

    # Try database lookup first
    try:
        book = pgdb.get_book_for_user(pod_id, user_id)
        if book and book.get("folder_path"):
            folder_path = pods_base / book["folder_path"]
            if folder_path.exists():
                return folder_path
    except Exception as e:
        logger.warning("Database lookup failed for pod_id=%s: %s", pod_id, e)

    # Fallback to legacy flat structure
    legacy_path = pods_base / pod_id
    if legacy_path.exists():
        return legacy_path

    # Try user-prefixed path
    user_path = pods_base / user_id / pod_id
    if user_path.exists():
        return user_path

    return None


def _generate_video(
    cover_path: Path,
    audio_path: Path,
    ass_path: Path,
    output_path: Path
) -> None:
    """Generate karaoke video using ffmpeg."""
    # Escape paths for ffmpeg filter (Windows paths need special handling)
    ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")

    filter_complex = (
        f"[0:v]scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},"
        f"drawbox=x=0:y=0:w={VIDEO_WIDTH}:h={VIDEO_HEIGHT}:color=black@0.8:t=fill,"
        f"ass='{ass_escaped}'[v]"
    )

    command = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(cover_path),
        "-i", str(audio_path),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "1:a",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-r", str(VIDEO_FPS),
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]

    logger.info("Running ffmpeg for video generation: %s", " ".join(command[:10]) + "...")
    t0 = time.time()

    try:
        proc = subprocess.run(command, check=True, capture_output=True, timeout=180)
        elapsed = time.time() - t0
        output_size = output_path.stat().st_size if output_path.exists() else 0
        logger.info("Video generated in %.1fs, size=%d bytes", elapsed, output_size)
        if proc.stderr:
            logger.debug("ffmpeg stderr: %s", proc.stderr.decode(errors="ignore")[-500:])
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
        logger.error("ffmpeg video generation failed: %s", stderr[-500:])
        raise RuntimeError(f"Video generation failed: {stderr.strip()}")
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg video generation timed out after 180s")
        raise RuntimeError("Video generation timed out")


def _generate_clip_video(
    words: List[Dict[str, Any]],
    clip_path: str,
    clip_start: float,
    pod_id: str,
    user_id: str,
    pgdb: Any
) -> Optional[str]:
    """Generate karaoke-style video for clip. Returns video URL or None."""
    if not words:
        logger.info("No words to generate video for")
        return None

    pods_base = Path(os.getenv("PODS_BASE", "")).expanduser()
    audio_url_prefix = os.getenv("AUDIO_URL_PREFIX", "")

    # Find book folder and cover image
    book_folder = _get_book_folder(pod_id, user_id, pgdb)
    if not book_folder:
        logger.warning("Book folder not found for pod_id=%s, user_id=%s", pod_id, user_id)
        return None

    cover_path = _find_cover_image(book_folder)
    if not cover_path:
        logger.warning("No cover image found in %s", book_folder)
        return None

    # Create clips output directory
    clips_dir = pods_base / user_id / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    video_id = str(uuid.uuid4())
    output_path = clips_dir / f"{video_id}.mp4"

    # Generate ASS subtitles
    ass_content = _generate_ass_subtitles(words, clip_start)

    # Write ASS to temp file
    ass_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".ass",
        delete=False,
        encoding="utf-8"
    )
    try:
        ass_file.write(ass_content)
        ass_file.close()
        ass_path = Path(ass_file.name)

        # Generate video
        _generate_video(
            cover_path=cover_path,
            audio_path=Path(clip_path),
            ass_path=ass_path,
            output_path=output_path
        )
    finally:
        Path(ass_file.name).unlink(missing_ok=True)

    # Generate URL
    video_url = f"{audio_url_prefix.rstrip('/')}/{user_id}/clips/{video_id}.mp4"
    logger.info("Video generated: %s", video_url)
    return video_url


def _process_clip(audio_url: str, timestamp: float, pod_id: str, user_id: str, pgdb: Any) -> Dict[str, Any]:
    logger.info("=== Clip generation started: url=%s, timestamp=%.1f ===", audio_url, timestamp)
    t0 = time.time()

    audio_source = _resolve_audio_path(audio_url)
    logger.info("Resolved audio source: %s", audio_source)

    clip_path = _extract_clip(audio_source, timestamp)

    try:
        words = _transcribe_clip(clip_path, timestamp)

        # Generate karaoke video
        video_url = None
        try:
            video_url = _generate_clip_video(
                words=words,
                clip_path=clip_path,
                clip_start=timestamp,
                pod_id=pod_id,
                user_id=user_id,
                pgdb=pgdb
            )
        except Exception as e:
            logger.warning("Video generation failed (continuing without video): %s", e)
            # Graceful degradation - return words without video

    finally:
        Path(clip_path).unlink(missing_ok=True)
        logger.debug("Temp file cleaned up: %s", clip_path)

    logger.info("=== Clip generation finished in %.1fs, %d words ===", time.time() - t0, len(words))
    return {
        "words": words,
        "clip_start": timestamp,
        "clip_end": timestamp + CLIP_DURATION_SECONDS,
        "video_url": video_url,
    }


@router.post("/api/clip", response_model=ClipResponse)
async def generate_clip(
    payload: ClipRequest,
    request: Request,
    session: SessionContainer = Depends(verify_session()),
) -> Dict[str, Any]:
    global _whisperx_lock
    if _whisperx_lock is None:
        _whisperx_lock = asyncio.Lock()

    user_id = session.get_user_id()
    pgdb = request.app.state.pgdb

    logger.info("POST /api/clip received: pod_id=%s, episode_idx=%d, timestamp=%.1f, audio_url=%s, user_id=%s",
                payload.pod_id, payload.episode_idx, payload.timestamp, payload.audio_url, user_id)

    if payload.timestamp > 86400:
        raise HTTPException(status_code=400, detail="Timestamp exceeds maximum.")

    lock_acquired = _whisperx_lock.locked()
    if lock_acquired:
        logger.info("Waiting for whisperx lock (another clip is being processed)...")

    async with _whisperx_lock:
        logger.info("Lock acquired, dispatching to thread...")
        try:
            result = await asyncio.to_thread(
                _process_clip,
                payload.audio_url,
                payload.timestamp,
                payload.pod_id,
                user_id,
                pgdb,
            )
        except RuntimeError as exc:
            logger.error("Clip generation RuntimeError: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))
        except Exception:
            logger.exception("Clip generation failed with unexpected error")
            raise HTTPException(status_code=500, detail="Clip generation failed.")

    logger.info("POST /api/clip returning %d words", len(result.get("words", [])))
    return result
