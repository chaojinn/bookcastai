from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont, ImageFilter

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

VIDEO_WIDTH = 406  # Must be even for H.264 (closest to ideal 405 for 9:16)
VIDEO_HEIGHT = 720
VIDEO_FPS = 24
VIDEO_CRF = 18  # Lower CRF preserves sharp text edges
VIDEO_TUNE = "stillimage"  # Optimize encoder for sharp static graphics
BACKGROUND_BLUR_RADIUS = 6  # Gaussian blur to soften cover text/details
MAX_CHARS_PER_LINE = 30
FONT_SIZE = 24
LINE_SPACING = 1.5  # Line height multiplier
LINE_HEIGHT_PX = int(FONT_SIZE * LINE_SPACING)  # 36px per line
VERTICAL_PADDING_RATIO = 0.10  # 10% padding on top and bottom

# Calculate visible lines based on usable vertical space
_usable_height = int(VIDEO_HEIGHT * (1 - 2 * VERTICAL_PADDING_RATIO))  # 80% of height
VISIBLE_LINES = _usable_height // LINE_HEIGHT_PX  # How many lines fit


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


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a system font with cross-platform fallback."""
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in font_paths:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # Fallback to default font
    return ImageFont.load_default()


def _prepare_background(cover_path: Path) -> Image.Image:
    """Prepare background image: cover scaled/cropped + 80% black overlay."""
    cover = Image.open(cover_path).convert("RGBA")

    # Scale to cover video dimensions (cover entire frame)
    cover_ratio = cover.width / cover.height
    video_ratio = VIDEO_WIDTH / VIDEO_HEIGHT

    if cover_ratio > video_ratio:
        # Cover is wider - scale by height, crop width
        new_height = VIDEO_HEIGHT
        new_width = int(VIDEO_HEIGHT * cover_ratio)
    else:
        # Cover is taller - scale by width, crop height
        new_width = VIDEO_WIDTH
        new_height = int(VIDEO_WIDTH / cover_ratio)

    cover = cover.resize((new_width, new_height), Image.LANCZOS)

    # Center crop to exact video dimensions
    left = (cover.width - VIDEO_WIDTH) // 2
    top = (cover.height - VIDEO_HEIGHT) // 2
    cover = cover.crop((left, top, left + VIDEO_WIDTH, top + VIDEO_HEIGHT))

    # Soft blur to de-emphasize cover text/details
    if BACKGROUND_BLUR_RADIUS > 0:
        cover = cover.filter(ImageFilter.GaussianBlur(radius=BACKGROUND_BLUR_RADIUS))

    # Apply 80% black overlay (204 = 0.8 * 255)
    overlay = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 204))
    cover = Image.alpha_composite(cover, overlay)

    return cover.convert("RGB")


def _render_frame(
    background: Image.Image,
    lines: List[List[Dict[str, Any]]],
    current_time: float,
    scroll_offset: int,
    font: ImageFont.FreeTypeFont
) -> Image.Image:
    """Render a single video frame with text highlighting."""
    frame = background.copy()
    draw = ImageDraw.Draw(frame)

    top_padding = int(VIDEO_HEIGHT * VERTICAL_PADDING_RATIO)
    bottom_limit = VIDEO_HEIGHT - top_padding

    for line_idx, line_words in enumerate(lines):
        if not line_words:
            continue

        line_y = top_padding + (line_idx * LINE_HEIGHT_PX) - scroll_offset

        # Skip lines outside visible area (with some margin for partial visibility)
        if line_y < top_padding - LINE_HEIGHT_PX or line_y > bottom_limit:
            continue

        # Render each word with appropriate color based on timing
        x_pos = 20  # Left margin
        for word in line_words:
            word_text = word.get("word", "")
            word_start = word.get("start", 0)
            word_end = word.get("end", 0)

            # Determine highlight state
            if current_time >= word_end:
                color = (255, 255, 255)  # White - fully spoken
            elif current_time >= word_start:
                color = (255, 255, 0)  # Yellow - currently speaking
            else:
                color = (128, 128, 128)  # Gray - not yet spoken

            draw.text((x_pos, line_y), word_text, font=font, fill=color)
            # Get text width for positioning next word
            bbox = draw.textbbox((0, 0), word_text + " ", font=font)
            x_pos += bbox[2] - bbox[0]

    return frame


def _generate_frames(
    lines: List[List[Dict[str, Any]]],
    clip_start: float,
    clip_duration: float,
    background: Image.Image,
    frames_dir: Path,
    font: ImageFont.FreeTypeFont
) -> int:
    """Generate all video frames as JPEG files. Returns frame count."""
    total_lines = len(lines)
    needs_scroll = total_lines > VISIBLE_LINES

    # Calculate scroll parameters
    scroll_trigger_line = VISIBLE_LINES // 2
    scroll_trigger_time = 0.0
    if needs_scroll and scroll_trigger_line < len(lines) and lines[scroll_trigger_line]:
        scroll_trigger_time = lines[scroll_trigger_line][0].get("start", clip_start) - clip_start

    extra_lines = max(0, total_lines - VISIBLE_LINES)
    max_scroll = extra_lines * LINE_HEIGHT_PX
    scroll_duration = clip_duration - scroll_trigger_time if scroll_trigger_time < clip_duration else 1.0

    total_frames = int(clip_duration * VIDEO_FPS)
    frame_count = 0

    logger.info("Generating %d frames (scroll=%s, trigger_time=%.1fs, max_scroll=%dpx)",
                total_frames, needs_scroll, scroll_trigger_time, max_scroll)

    for frame_idx in range(total_frames):
        current_time = clip_start + (frame_idx / VIDEO_FPS)
        elapsed = frame_idx / VIDEO_FPS

        # Calculate scroll offset
        scroll_offset = 0
        if needs_scroll and elapsed > scroll_trigger_time and scroll_duration > 0:
            scroll_progress = (elapsed - scroll_trigger_time) / scroll_duration
            scroll_progress = min(1.0, max(0.0, scroll_progress))
            scroll_offset = int(scroll_progress * max_scroll)

        # Render frame
        frame = _render_frame(background, lines, current_time, scroll_offset, font)

        # Save frame as JPEG (quality 85 keeps size < 100KB)
        frame_path = frames_dir / f"frame_{frame_idx:05d}.jpg"
        frame.save(frame_path, "JPEG", quality=85)
        frame_count += 1

        # Log progress every 10%
        if frame_idx > 0 and frame_idx % (total_frames // 10) == 0:
            logger.debug("Frame generation progress: %d/%d (%.0f%%)",
                        frame_idx, total_frames, 100 * frame_idx / total_frames)

    logger.info("Generated %d frames in %s", frame_count, frames_dir)
    return frame_count


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
    frames_dir: Path,
    audio_path: Path,
    output_path: Path,
    fps: int = VIDEO_FPS
) -> None:
    """Generate video from JPEG frames and audio using ffmpeg."""
    frame_pattern = str(frames_dir / "frame_%05d.jpg")

    command = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", str(VIDEO_CRF),
        "-tune", VIDEO_TUNE,
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]

    logger.info("Running ffmpeg for video generation: %s", " ".join(command[:10]) + "...")
    t0 = time.time()

    try:
        proc = subprocess.run(command, check=True, capture_output=True, timeout=300)
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
        logger.error("ffmpeg video generation timed out after 300s")
        raise RuntimeError("Video generation timed out")


def _generate_clip_video(
    words: List[Dict[str, Any]],
    clip_path: str,
    clip_start: float,
    pod_id: str,
    user_id: str,
    pgdb: Any
) -> Optional[str]:
    """Generate karaoke-style video for clip using frame-by-frame rendering.

    Returns video URL or None.
    """
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
    frames_dir = clips_dir / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_path = clips_dir / f"{video_id}.mp4"

    logger.info("Generating video: frames_dir=%s, output=%s", frames_dir, output_path)

    # Prepare background (cover + overlay)
    background = _prepare_background(cover_path)

    # Load font
    font = _load_font(FONT_SIZE)

    # Group words into lines
    lines = _group_words_into_lines(words)
    logger.info("Grouped %d words into %d lines", len(words), len(lines))

    # Generate all frames
    t0 = time.time()
    frame_count = _generate_frames(
        lines=lines,
        clip_start=clip_start,
        clip_duration=CLIP_DURATION_SECONDS,
        background=background,
        frames_dir=frames_dir,
        font=font
    )
    logger.info("Frame generation took %.1fs for %d frames", time.time() - t0, frame_count)

    # Generate video from frames
    _generate_video(
        frames_dir=frames_dir,
        audio_path=Path(clip_path),
        output_path=output_path
    )

    # Cleanup frames after successful video generation
    shutil.rmtree(frames_dir, ignore_errors=True)
    logger.debug("Cleaned up frames directory: %s", frames_dir)

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
