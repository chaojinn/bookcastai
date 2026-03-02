from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    import numpy as np

from tts.tts_provider import TTSProvider, TTSProviderError, TTSRequest

INTERNAL_SPEAKERS = ("Serena", "Vivian", "Aiden", "Ryan")
BASE_MODEL_ID = "Qwen/Qwen3-TTS"
DEFAULT_BATCH_SIZE = 4  # chunks processed per GPU call; increase if VRAM allows
SPEAKERS_DIR = Path(__file__).parent / "speakers"
SUPPORTED_FORMATS = {"mp3", "wav"}

logger = logging.getLogger(__name__)

_QWEN3_MODEL_CLS = None
_SF = None
_AUDIO_SEGMENT = None


def _ensure_qwen3():
    global _QWEN3_MODEL_CLS
    if _QWEN3_MODEL_CLS is None:
        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore
        except ImportError as exc:
            raise TTSProviderError(
                "The 'qwen_tts' package is not installed. Install it to use Qwen3TTSProvider."
            ) from exc
        _QWEN3_MODEL_CLS = Qwen3TTSModel
    return _QWEN3_MODEL_CLS


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
        except ImportError:
            return None
        _AUDIO_SEGMENT = AudioSegment
    return _AUDIO_SEGMENT


def _load_model(path_or_id: str, device: str):
    Qwen3TTSModel = _ensure_qwen3()
    import torch  # type: ignore

    try:
        import flash_attn  # noqa: F401  # type: ignore
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    try:
        model = Qwen3TTSModel.from_pretrained(
            path_or_id,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
    except Exception as exc:
        raise TTSProviderError(f"Failed to load Qwen3-TTS model from '{path_or_id}': {exc}") from exc
    model.model.eval()
    return model


def _split_sentences(text: str, max_chars: int = 300) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    raw = re.split(r"(?<=[.!?…])\s+", text)
    chunks: list[str] = []
    current = ""
    for sent in raw:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            current = ""
            if len(sent) > max_chars:
                parts = re.split(r"(?<=,)\s+", sent)
                sub = ""
                for part in parts:
                    if len(sub) + len(part) + 1 <= max_chars:
                        sub = (sub + " " + part).strip()
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = part
                if sub:
                    chunks.append(sub)
            else:
                current = sent
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def _wav_to_numpy(w) -> "np.ndarray":
    import numpy as np  # type: ignore

    if isinstance(w, np.ndarray):
        arr = w
    elif hasattr(w, "cpu"):
        arr = w.cpu().float().numpy()
    else:
        arr = np.array(w, dtype=np.float32)
    arr = arr.squeeze()
    if arr.ndim != 1:
        arr = arr.flatten()
    return arr.astype(np.float32)


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


def _write_wav(path: Path, waveform: "np.ndarray", sample_rate: int) -> None:
    sf = _ensure_soundfile()
    sf.write(str(path), waveform, sample_rate, format="WAV", subtype="PCM_16")


def _write_mp3(path: Path, waveform: "np.ndarray", sample_rate: int) -> None:
    if _ffmpeg_available():
        _encode_mp3_with_ffmpeg(path, waveform, sample_rate)
        return

    AudioSegment = _ensure_pydub()
    if AudioSegment is None:
        raise TTSProviderError(
            "pydub is required to export MP3 audio but is not installed, and ffmpeg is unavailable. "
            "Install pydub or ffmpeg."
        )
    sf = _ensure_soundfile()
    buffer = BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    try:
        audio_segment = AudioSegment.from_file(buffer, format="wav")
    except Exception as exc:
        raise TTSProviderError(f"Failed to load generated audio for MP3 encoding: {exc}") from exc
    try:
        audio_segment.export(path, format="mp3")
    except Exception as exc:
        raise TTSProviderError(
            f"Failed to write MP3 audio to '{path}'. Ensure ffmpeg is installed and on PATH. ({exc})"
        ) from exc


def _encode_mp3_with_ffmpeg(path: Path, waveform: "np.ndarray", sample_rate: int) -> None:
    sf = _ensure_soundfile()
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav_path = Path(tmp_wav.name)
    try:
        sf.write(tmp_wav_path, waveform, sample_rate, format="WAV", subtype="PCM_16")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(tmp_wav_path),
            "-ar", str(sample_rate),
            "-ac", "1",
            "-c:a", "libmp3lame",
            "-b:a", "64k",
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


class Qwen3TTSProvider(TTSProvider):
    """Generate speech with the Qwen3-TTS 1.7B model.

    Internal speakers (Serena, Vivian, Aiden, Ryan) use the base HuggingFace
    model. Custom speakers are loaded from fine-tuned checkpoints stored as
    subfolders under tts/qwen3/speakers/.

    Supported parameters (via TTSRequest.raw_parameters JSON):
        speaker   (str)   – voice name; default "Serena"
        temperature (float) – sampling temperature (if supported by model)
        format    (str)   – "wav" or "mp3"; default inferred from output_file suffix
    """

    def __init__(
        self,
        base_model_id: str = BASE_MODEL_ID,
        device: str = "cuda:0",
    ) -> None:
        self.base_model_id = base_model_id
        self.device = device
        self._base_model = None
        self._custom_models: Dict[str, object] = {}

    def get_english_voices(self) -> list[dict[str, str]]:
        voices = [{"name": s, "code": s} for s in INTERNAL_SPEAKERS]
        if SPEAKERS_DIR.exists():
            for folder in sorted(SPEAKERS_DIR.iterdir()):
                if folder.is_dir():
                    voices.append({"name": folder.name, "code": folder.name})
        return voices

    def tts(self, request: TTSRequest) -> Path:
        import numpy as np  # type: ignore
        import torch  # type: ignore

        params = request.parsed_parameters()
        speaker_val = params.get("speaker", INTERNAL_SPEAKERS[0])
        speaker = str(speaker_val) if speaker_val else INTERNAL_SPEAKERS[0]
        temperature_val = params.get("temperature")
        fmt = str(params.get("format", request.output_file.suffix.lstrip(".") or "wav")).lower()

        if fmt not in SUPPORTED_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_FORMATS))
            raise TTSProviderError(f"Unsupported audio format '{fmt}'. Supported formats: {supported}.")

        is_internal = speaker in INTERNAL_SPEAKERS
        model = self._get_base_model() if is_internal else self._get_custom_model(speaker)

        chunks = _split_sentences(request.text_content)
        if not chunks:
            raise TTSProviderError("No text content to synthesise.")

        batch_size_val = params.get("batch_size", DEFAULT_BATCH_SIZE)
        try:
            batch_size = max(1, int(batch_size_val))
        except (TypeError, ValueError):
            batch_size = DEFAULT_BATCH_SIZE

        gen_kwargs: dict = {"speaker": speaker}
        if temperature_val is not None:
            try:
                gen_kwargs["temperature"] = float(temperature_val)
            except (TypeError, ValueError) as exc:
                raise TTSProviderError(f"Invalid temperature value: {temperature_val}") from exc

        all_audio: list[np.ndarray] = []
        sample_rate: int | None = None

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start:batch_start + batch_size]
            batch_end = batch_start + len(batch)
            logger.debug(
                "Qwen3 batch [%d-%d/%d]: first=%r",
                batch_start + 1, batch_end, len(chunks), batch[0][:70],
            )
            try:
                with torch.inference_mode():
                    wavs, sr = model.generate_custom_voice(text=batch, **gen_kwargs)
            except Exception as exc:
                raise TTSProviderError(
                    f"Qwen3-TTS synthesis failed on chunks {batch_start + 1}-{batch_end}: {exc}"
                ) from exc
            sample_rate = sr
            for wav in wavs:
                all_audio.append(_wav_to_numpy(wav))

        if not all_audio:
            raise TTSProviderError("Qwen3-TTS generated no audio output.")

        combined = np.concatenate(all_audio)

        if fmt == "mp3":
            _write_mp3(request.output_file, combined, sample_rate)
        else:
            _write_wav(request.output_file, combined, sample_rate)

        return request.output_file

    def _get_base_model(self):
        if self._base_model is None:
            logger.info("Loading Qwen3-TTS base model from %s", self.base_model_id)
            self._base_model = _load_model(self.base_model_id, self.device)
        return self._base_model

    def _get_custom_model(self, speaker: str):
        if speaker not in self._custom_models:
            speaker_path = SPEAKERS_DIR / speaker
            if not speaker_path.exists():
                available = [d.name for d in SPEAKERS_DIR.iterdir() if d.is_dir()] if SPEAKERS_DIR.exists() else []
                raise TTSProviderError(
                    f"Custom speaker '{speaker}' not found in {SPEAKERS_DIR}. "
                    f"Available: {available or 'none'}."
                )
            logger.info("Loading Qwen3-TTS custom model for speaker '%s'", speaker)
            self._custom_models[speaker] = _load_model(str(speaker_path), self.device)
        return self._custom_models[speaker]
