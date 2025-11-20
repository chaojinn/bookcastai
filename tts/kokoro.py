from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf
from io import BytesIO

try:
    from pydub import AudioSegment
except ImportError:  # pragma: no cover - handled at runtime
    AudioSegment = None  # type: ignore

try:
    from kokoro import KPipeline
    from kokoro.pipeline import ALIASES as KOKORO_ALIASES, LANG_CODES as KOKORO_LANG_CODES
except ImportError:  # pragma: no cover - handled at runtime
    KPipeline = None  # type: ignore
    KOKORO_ALIASES = {}
    KOKORO_LANG_CODES = {}

from tts.tts_provider import TTSProvider, TTSProviderError, TTSRequest

DEFAULT_VOICE = "af_bella"
DEFAULT_LANG = "en-us"
DEFAULT_SPEED = 0.9
SUPPORTED_FORMATS = {"mp3", "wav"}
DEFAULT_SAMPLE_RATE = 24_000

logger = logging.getLogger(__name__)


class KokoroTTSProvider(TTSProvider):
    """Generate speech with the Kokoro-82M model via the kokoro Python library."""

    def __init__(
        self,
        executable: str = "kokoro-tts",
        *,
        device: str | None = None,
        repo_id: str = "hexgrad/Kokoro-82M",
    ) -> None:
        self.executable = executable  # Retained for backwards compatibility with existing configs.
        self.device = device
        self.repo_id = repo_id
        self._pipelines: Dict[str, KPipeline] = {}

    def tts(self, request: TTSRequest) -> Path:
        if KPipeline is None:
            raise TTSProviderError(
                "The 'kokoro' package is not installed. Install it via 'pip install kokoro>=0.9.4'."
            )

        params = request.parsed_parameters()
        voice_value = params.get("voice", DEFAULT_VOICE)
        lang_value = params.get("lang", DEFAULT_LANG)
        voice = str(voice_value) if voice_value else DEFAULT_VOICE
        lang = str(lang_value) if lang_value else DEFAULT_LANG
        speed = params.get("speed", DEFAULT_SPEED)
        fmt = str(params.get("format", request.output_file.suffix.lstrip(".") or "mp3")).lower()
        split_pattern_value = params.get("split_pattern")
        split_pattern = split_pattern_value if isinstance(split_pattern_value, str) else "$^"

        if fmt not in SUPPORTED_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_FORMATS))
            raise TTSProviderError(f"Unsupported audio format '{fmt}'. Supported formats: {supported}.")

        try:
            speed_value = float(speed)
        except (TypeError, ValueError) as exc:
            raise TTSProviderError(f"Invalid speed value: {speed}") from exc
        if speed_value <= 0:
            raise TTSProviderError("Speed must be a positive value.")

        lang_code = self._resolve_lang_code(lang, voice)
        pipeline = self._get_pipeline(lang_code)

        pipeline_kwargs = {"voice": voice, "speed": speed_value, "split_pattern": split_pattern}
        logger.info("Kokoro split_pattern in use: %s", split_pattern)

        try:
            generator = pipeline(
                request.text_content,
                **pipeline_kwargs,
            )
        except Exception as exc:  # pragma: no cover - kokoro raises rich torch errors
            raise TTSProviderError(f"Kokoro synthesis failed to start: {exc}") from exc

        wave_chunks = []
        for result in generator:
            audio = getattr(result, "audio", None)
            if audio is None:
                continue
            wave_chunks.append(audio.detach().cpu().numpy().astype(np.float32, copy=False))

        if not wave_chunks:
            raise TTSProviderError("Kokoro generated no audio output for the supplied text.")

        waveform = np.concatenate(wave_chunks).astype(np.float32)
        waveform = np.clip(waveform, -1.0, 1.0)

        if fmt == "wav":
            self._write_wav(request.output_file, waveform)
        else:
            self._write_mp3(request.output_file, waveform)

        return request.output_file

    def _resolve_lang_code(self, lang: str, voice: str) -> str:
        normalized_lang = lang.replace("_", "-").lower()
        if normalized_lang in KOKORO_ALIASES:
            return KOKORO_ALIASES[normalized_lang]
        voice_prefix = voice.split("_", 1)[0].lower() if voice else ""
        if voice_prefix:
            candidate = voice_prefix[:1]
            if candidate in KOKORO_LANG_CODES:
                return candidate
        if normalized_lang in KOKORO_LANG_CODES:
            return normalized_lang
        supported = ", ".join(sorted(KOKORO_LANG_CODES))
        raise TTSProviderError(
            f"Could not determine Kokoro language code for lang='{lang}' and voice='{voice}'. "
            f"Supported language codes: {supported}."
        )

    def _get_pipeline(self, lang_code: str) -> KPipeline:
        if lang_code in self._pipelines:
            return self._pipelines[lang_code]
        try:
            pipeline = KPipeline(lang_code=lang_code, repo_id=self.repo_id, device=self.device)
        except Exception as exc:  # pragma: no cover - kokoro raises rich torch errors
            raise TTSProviderError(f"Failed to initialise Kokoro pipeline for '{lang_code}': {exc}") from exc
        self._pipelines[lang_code] = pipeline
        return pipeline

    @staticmethod
    def _write_wav(path: Path, waveform: np.ndarray) -> None:
        sf.write(str(path), waveform, DEFAULT_SAMPLE_RATE, format="WAV", subtype="PCM_16")

    @staticmethod
    def _write_mp3(path: Path, waveform: np.ndarray) -> None:
        if _ffmpeg_available():
            _encode_mp3_with_ffmpeg(path, waveform)
            return

        if AudioSegment is None:
            raise TTSProviderError(
                "pydub is required to export MP3 audio but is not installed, and ffmpeg is unavailable. "
                "Install pydub or ffmpeg."
            )
        buffer = BytesIO()
        sf.write(buffer, waveform, DEFAULT_SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        try:
            audio_segment = AudioSegment.from_file(buffer, format="wav")
        except Exception as exc:  # pragma: no cover - dependent on ffmpeg availability
            raise TTSProviderError(f"Failed to load generated audio for MP3 encoding: {exc}") from exc
        try:
            audio_segment.export(path, format="mp3")
        except Exception as exc:  # pragma: no cover - dependent on ffmpeg availability
            raise TTSProviderError(
                f"Failed to write MP3 audio to '{path}'. Ensure ffmpeg is installed and on PATH. ({exc})"
            ) from exc


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-version"],
            check=True,
            capture_output=True,
        )
        return True
    except Exception:
        return False


def _encode_mp3_with_ffmpeg(path: Path, waveform: np.ndarray) -> None:
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav_path = Path(tmp_wav.name)
    try:
        sf.write(tmp_wav_path, waveform, DEFAULT_SAMPLE_RATE, format="WAV", subtype="PCM_16")
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(tmp_wav_path),
            "-ar",
            str(DEFAULT_SAMPLE_RATE),
            "-ac",
            "1",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "64k",
            str(path),
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise TTSProviderError(f"ffmpeg MP3 encoding failed: {exc}") from exc
    finally:
        try:
            tmp_wav.close()
            tmp_wav_path.unlink(missing_ok=True)
        except Exception:
            logger.debug("Temporary WAV cleanup failed for %s", tmp_wav_path)
