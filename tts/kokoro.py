from __future__ import annotations

import logging
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    import numpy as np
    from kokoro import KPipeline

from tts.tts_provider import TTSProvider, TTSProviderError, TTSRequest

DEFAULT_VOICE = "af_bella"
DEFAULT_LANG = "en-us"
DEFAULT_SPEED = 0.9
SUPPORTED_FORMATS = {"mp3", "wav"}
DEFAULT_SAMPLE_RATE = 24_000
ENGLISH_VOICE_NAMES = (
    "American Female Heart",
    "American Female Alloy",
    "American Female Aoede",
    "American Female Bella",
    "American Female Jessica",
    "American Female Kore",
    "American Female Nicole",
    "American Female Nova",
    "American Female River",
    "American Female Sarah",
    "American Female Sky",
    "American Male Adam",
    "American Male Echo",
    "American Male Eric",
    "American Male Fenrir",
    "American Male Liam",
    "American Male Michael",
    "American Male Onyx",
    "American Male Puck",
    "American Male Santa",
    "British Female Alice",
    "British Female Emma",
    "British Female Isabella",
    "British Female Lily",
    "British Male Daniel",
    "British Male Fable",
    "British Male George",
    "British Male Lewis",
)
ENGLISH_VOICE_CODES = (
    "af_heart",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
)

logger = logging.getLogger(__name__)

_NP = None
_SF = None
_AUDIO_SEGMENT = None
_KPIPELINE = None
_KOKORO_ALIASES = None
_KOKORO_LANG_CODES = None


def _ensure_numpy():
    global _NP
    if _NP is None:
        import numpy as np  # type: ignore

        _NP = np
    return _NP


def _ensure_soundfile():
    global _SF
    if _SF is None:
        import soundfile as sf  # type: ignore

        _SF = sf
    return _SF


def _ensure_pydub():
    global _AUDIO_SEGMENT
    if _AUDIO_SEGMENT is None:
        try:
            from pydub import AudioSegment  # type: ignore
        except ImportError:  # pragma: no cover - handled at runtime
            return None
        _AUDIO_SEGMENT = AudioSegment
    return _AUDIO_SEGMENT


def _ensure_kokoro():
    global _KPIPELINE, _KOKORO_ALIASES, _KOKORO_LANG_CODES
    if _KPIPELINE is None or _KOKORO_ALIASES is None or _KOKORO_LANG_CODES is None:
        try:
            from kokoro import KPipeline  # type: ignore
            from kokoro.pipeline import ALIASES, LANG_CODES  # type: ignore
        except ImportError as exc:  # pragma: no cover - handled at runtime
            raise TTSProviderError(
                "The 'kokoro' package is not installed. Install it via 'pip install kokoro>=0.9.4'."
            ) from exc
        _KPIPELINE = KPipeline
        _KOKORO_ALIASES = ALIASES
        _KOKORO_LANG_CODES = LANG_CODES
    return _KPIPELINE, _KOKORO_ALIASES, _KOKORO_LANG_CODES


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
        _ensure_kokoro()
        np = _ensure_numpy()

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
        _, kokoro_aliases, kokoro_lang_codes = _ensure_kokoro()
        normalized_lang = lang.replace("_", "-").lower()
        if normalized_lang in kokoro_aliases:
            return kokoro_aliases[normalized_lang]
        voice_prefix = voice.split("_", 1)[0].lower() if voice else ""
        if voice_prefix:
            candidate = voice_prefix[:1]
            if candidate in kokoro_lang_codes:
                return candidate
        if normalized_lang in kokoro_lang_codes:
            return normalized_lang
        supported = ", ".join(sorted(kokoro_lang_codes))
        raise TTSProviderError(
            f"Could not determine Kokoro language code for lang='{lang}' and voice='{voice}'. "
            f"Supported language codes: {supported}."
        )

    def _get_pipeline(self, lang_code: str) -> "KPipeline":
        KPipeline, _, _ = _ensure_kokoro()
        if lang_code in self._pipelines:
            return self._pipelines[lang_code]
        try:
            pipeline = KPipeline(lang_code=lang_code, repo_id=self.repo_id, device=self.device)
        except Exception as exc:  # pragma: no cover - kokoro raises rich torch errors
            raise TTSProviderError(f"Failed to initialise Kokoro pipeline for '{lang_code}': {exc}") from exc
        self._pipelines[lang_code] = pipeline
        return pipeline

    def get_english_voices(self) -> list[dict[str, str]]:
        return [
            {"name": name, "code": code}
            for name, code in zip(ENGLISH_VOICE_NAMES, ENGLISH_VOICE_CODES, strict=True)
        ]

    @staticmethod
    def _write_wav(path: Path, waveform: "np.ndarray") -> None:
        sf = _ensure_soundfile()
        sf.write(str(path), waveform, DEFAULT_SAMPLE_RATE, format="WAV", subtype="PCM_16")

    @staticmethod
    def _write_mp3(path: Path, waveform: "np.ndarray") -> None:
        if _ffmpeg_available():
            _encode_mp3_with_ffmpeg(path, waveform)
            return

        AudioSegment = _ensure_pydub()
        if AudioSegment is None:
            raise TTSProviderError(
                "pydub is required to export MP3 audio but is not installed, and ffmpeg is unavailable. "
                "Install pydub or ffmpeg."
            )
        sf = _ensure_soundfile()
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


def _encode_mp3_with_ffmpeg(path: Path, waveform: "np.ndarray") -> None:
    sf = _ensure_soundfile()
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
