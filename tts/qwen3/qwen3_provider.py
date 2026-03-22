from __future__ import annotations

import gc
import logging
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict


if TYPE_CHECKING:
    import numpy as np

from tts.tts_provider import TTSProvider, TTSProviderError, TTSRequest

INTERNAL_SPEAKERS = ("Serena", "Vivian", "Aiden", "Ryan")
BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_BATCH_SIZE = 8  # chunks processed per GPU call; increase if VRAM allows
SPEAKERS_DIR = Path(__file__).parent / "speakers"
SUPPORTED_FORMATS = {"mp3", "wav"}
CHUNK_SILENCE_PAD_MS = 1000  # silence inserted between consecutive TTS chunks

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


def _patch_deepcopy_for_dict_keys() -> None:
    """Register a deepcopy handler for dict_keys so bitsandbytes quantization works.

    transformers calls deepcopy(model) inside get_keys_to_not_convert() when
    preparing 8-bit quantization.  If the model state contains dict_keys values
    (e.g. tied-weights mappings) the default deepcopy raises
    ``TypeError: cannot pickle 'dict_keys' object``.
    Converting dict_keys → list is always correct for a deepcopy.
    """
    import copy
    dict_keys_type = type({}.keys())
    if dict_keys_type not in copy._deepcopy_dispatch:
        copy._deepcopy_dispatch[dict_keys_type] = lambda x, memo: list(x)


def _load_model(path_or_id: str, device: str, *, quantize: bool = False, warmup_speaker: str | None = None, tts_log: "_TtsLog | None" = None):
    Qwen3TTSModel = _ensure_qwen3()
    import torch  # type: ignore

    try:
        import flash_attn  # noqa: F401  # type: ignore
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    kwargs: dict = {
        "device_map": device,
        "attn_implementation": attn_impl,
    }
    if quantize:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
            _patch_deepcopy_for_dict_keys()
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except ImportError as exc:
            raise TTSProviderError(
                "INT8 quantization requires the 'bitsandbytes' package. "
                "Install it with: pip install bitsandbytes"
            ) from exc
    else:
        kwargs["dtype"] = torch.bfloat16

    precision = "INT8 (8-bit)" if quantize else "bfloat16 (16-bit)"
    print(f"[Qwen3TTS] Loading model '{path_or_id}' on {device} in {precision}", flush=True)
    try:
        model = Qwen3TTSModel.from_pretrained(path_or_id, **kwargs)
    except Exception as exc:
        raise TTSProviderError(f"Failed to load Qwen3-TTS model from '{path_or_id}': {exc}") from exc
    print(f"[Qwen3TTS] Model loaded successfully in {precision}", flush=True)
    model.model.eval()

    # Warmup: discard the first inference to avoid garbage audio caused by
    # CUDA JIT compilation and uninitialized internal model state.
    speaker = warmup_speaker or INTERNAL_SPEAKERS[0]
    print(f"[Qwen3TTS] Running warmup inference (speaker={speaker})...", flush=True)
    if tts_log is not None:
        tts_log.warmup_start(speaker)
    import torch as _torch  # type: ignore
    import gc as _gc
    with _torch.inference_mode():
        model.generate_custom_voice(text=["Hello."], speaker=speaker)
    _gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.synchronize()
        _torch.cuda.empty_cache()
    print("[Qwen3TTS] Warmup done.", flush=True)
    if tts_log is not None:
        tts_log.warmup_end()

    return model



def _trim_silence(audio: "np.ndarray", sample_rate: int, threshold_db: float = -40.0) -> "np.ndarray":
    """Trim leading and trailing silence below threshold_db from a waveform."""
    import numpy as np  # type: ignore
    threshold = 10 ** (threshold_db / 20.0)
    nonsilent = np.where(np.abs(audio) > threshold)[0]
    if len(nonsilent) == 0:
        return audio
    return audio[nonsilent[0]:nonsilent[-1] + 1]


def _compress_long_silences(
    audio: "np.ndarray",
    sample_rate: int,
    max_silence_sec: float = 2.0,
    threshold_db: float = -20.0,
) -> "tuple[np.ndarray, list[tuple[float, float]]]":
    """Cap any contiguous silent region longer than max_silence_sec to exactly max_silence_sec.

    Returns the processed audio and a list of (start_sec, end_sec) tuples for each
    silence region that was compressed.
    """
    import numpy as np  # type: ignore

    threshold = 10 ** (threshold_db / 20.0)
    max_samples = int(sample_rate * max_silence_sec)
    is_silent = np.abs(audio) <= threshold

    segments: list[np.ndarray] = []
    compressed: list[tuple[float, float]] = []
    i = 0
    n = len(audio)
    while i < n:
        if is_silent[i]:
            j = i
            while j < n and is_silent[j]:
                j += 1
            silence_len = j - i
            keep = min(silence_len, max_samples)
            segments.append(audio[i : i + keep])
            if silence_len > max_samples:
                compressed.append((i / sample_rate, j / sample_rate))
            i = j
        else:
            j = i
            while j < n and not is_silent[j]:
                j += 1
            segments.append(audio[i:j])
            i = j

    result = np.concatenate(segments) if segments else audio
    return result, compressed


class _TtsLog:
    """Append-mode log writer for a single TTS task."""

    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = log_path.open("a", encoding="utf-8")
        self._write_line("")  # blank separator between runs

    def _write_line(self, msg: str) -> None:
        import datetime
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._f.write(f"[{ts}] {msg}\n")
        self._f.flush()

    def chapter_start(self, chapter: str) -> None:
        self._write_line(f"=== Starting chapter: {chapter} ===")

    def chapter_end(self, chapter: str) -> None:
        self._write_line(f"=== End chapter: {chapter} ===")

    def chunk_start(self, index: int, text: str) -> None:
        single_line = " ".join(text.split())
        self._write_line(f"Chunk {index} start: {single_line}")

    def chunk_end(self, index: int) -> None:
        self._write_line(f"Chunk {index} end")

    def chunk_silence_compressed(self, index: int, start_sec: float, end_sec: float) -> None:
        self._write_line(
            f"Chunk {index}: silence at {start_sec:.2f}s-{end_sec:.2f}s compressed to 1.00s"
        )

    def batch_start(self, batch_start: int, batch_end: int, total: int) -> None:
        self._write_line(f"Batch [{batch_start + 1}-{batch_end}/{total}] generating...")

    def batch_end(self, batch_start: int, batch_end: int, total: int) -> None:
        self._write_line(f"Batch [{batch_start + 1}-{batch_end}/{total}] done")

    def concat_start(self, n_chunks: int) -> None:
        self._write_line(f"Concatenating {n_chunks} chunks...")

    def concat_end(self) -> None:
        self._write_line("Concatenation done")

    def encode_start(self, fmt: str, path: str) -> None:
        self._write_line(f"Encoding to {fmt}: {path}")

    def encode_end(self) -> None:
        self._write_line("Encoding done")

    def warmup_start(self, speaker: str) -> None:
        self._write_line(f"--- Warmup start (speaker={speaker}) ---")

    def warmup_end(self) -> None:
        self._write_line("--- Warmup end ---")

    def close(self) -> None:
        self._f.close()


def _silence_pad(ms: int, sample_rate: int) -> "np.ndarray":
    """Return an array of silence of the given duration in milliseconds."""
    import numpy as np  # type: ignore
    n_samples = int(sample_rate * ms / 1000)
    return np.zeros(n_samples, dtype=np.float32)


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
        speaker     (str)   – voice name; default "Serena"
        temperature (float) – sampling temperature (if supported by model)
        format      (str)   – "wav" or "mp3"; default inferred from output_file suffix
        seed        (int)   – random seed for deterministic generation; default 42
        quantize    (bool)  – load model in INT8 to halve VRAM; batch_size is doubled automatically; default false
    """

    def __init__(
        self,
        base_model_id: str = BASE_MODEL_ID,
        device: str = "cuda:0",
    ) -> None:
        self.base_model_id = base_model_id
        self.device = device
        self._base_model = None
        self._custom_models: Dict[tuple, object] = {}

    def get_english_voices(self) -> list[dict[str, str]]:
        voices = [{"name": s, "code": s} for s in INTERNAL_SPEAKERS]
        if SPEAKERS_DIR.exists():
            for folder in sorted(SPEAKERS_DIR.iterdir()):
                if folder.is_dir():
                    voices.append({"name": folder.name, "code": folder.name})
        return voices

    def tts(
        self,
        request: TTSRequest,
        *,
        chunk_progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
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

        quantize = bool(params.get("quantize", False))
        is_internal = speaker in INTERNAL_SPEAKERS

        tts_log = _TtsLog(request.output_file.parent / "debug" / "tts.log")

        model = self._get_base_model(quantize=quantize, tts_log=tts_log) if is_internal else self._get_custom_model(speaker, quantize=quantize, tts_log=tts_log)

        from agent.chunkrizer import chunk_text  # lazy import to avoid module-level side effects
        chunks = chunk_text(request.text_content, 500)
        if not chunks:
            raise TTSProviderError("No text content to synthesise.")

        batch_size_val = params.get("batch_size", DEFAULT_BATCH_SIZE * 2 if quantize else DEFAULT_BATCH_SIZE)
        try:
            batch_size = max(1, int(batch_size_val))
        except (TypeError, ValueError):
            batch_size = DEFAULT_BATCH_SIZE * 2 if quantize else DEFAULT_BATCH_SIZE

        gen_kwargs: dict = {"speaker": speaker, "max_new_tokens": 1440}  # 1440 tokens @ 12Hz = 120s max per chunk
        if temperature_val is not None:
            try:
                gen_kwargs["temperature"] = float(temperature_val)
            except (TypeError, ValueError) as exc:
                raise TTSProviderError(f"Invalid temperature value: {temperature_val}") from exc

        seed_val = params.get("seed", 42)
        try:
            seed = int(seed_val)
        except (TypeError, ValueError):
            seed = 42
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        all_audio: list[np.ndarray] = []
        sample_rate: int | None = None

        debug_dir = request.output_file.parent / "debug" / request.output_file.stem
        debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving debug chunk WAVs to %s", debug_dir)

        tts_log.chapter_start(request.output_file.stem)

        chunk_index = 0
        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start:batch_start + batch_size]
            batch_end = batch_start + len(batch)
            tts_log.batch_start(batch_start, batch_end, len(chunks))
            try:
                with torch.inference_mode():
                    wavs, sr = model.generate_custom_voice(text=batch, **gen_kwargs)
                # Move all outputs to CPU immediately so GPU memory is freed
                # before the next batch's generation begins.
                if hasattr(wavs, "cpu"):
                    wavs = wavs.cpu()
                elif isinstance(wavs, (list, tuple)):
                    wavs = [w.cpu() if hasattr(w, "cpu") else w for w in wavs]
            except Exception as exc:
                tts_log.close()
                raise TTSProviderError(
                    f"Qwen3-TTS synthesis failed on chunks {batch_start + 1}-{batch_end}: {exc}"
                ) from exc
            sample_rate = sr
            tts_log.batch_end(batch_start, batch_end, len(chunks))
            torch.cuda.empty_cache()
            for wav, chunk_text in zip(wavs, batch):
                tts_log.chunk_start(chunk_index, chunk_text)
                arr = _wav_to_numpy(wav)
                arr = _trim_silence(arr, sample_rate)
                #arr, compressed = _compress_long_silences(arr, sample_rate)
                compressed = []
                for start_sec, end_sec in compressed:
                    tts_log.chunk_silence_compressed(chunk_index, start_sec, end_sec)
                chunk_path = debug_dir / f"chunk_{chunk_index:04d}.wav"
                _write_wav(chunk_path, arr, sample_rate)
                logger.debug("Saved chunk %d to %s (%.2fs)", chunk_index, chunk_path, len(arr) / sample_rate)
                tts_log.chunk_end(chunk_index)
                all_audio.append(arr)
                all_audio.append(_silence_pad(CHUNK_SILENCE_PAD_MS, sample_rate))
                chunk_index += 1
            del wavs
            gc.collect()
            torch.cuda.empty_cache()
            if chunk_progress_callback is not None:
                chunk_progress_callback(batch_end, len(chunks))

        if not all_audio:
            tts_log.close()
            raise TTSProviderError("Qwen3-TTS generated no audio output.")

        # Drop the trailing 200ms pad appended after the last chunk
        if len(all_audio) >= 2:
            all_audio = all_audio[:-1]

        tts_log.concat_start(chunk_index)
        combined = np.concatenate(all_audio)
        del all_audio
        tts_log.concat_end()

        tts_log.encode_start(fmt, str(request.output_file))
        if fmt == "mp3":
            _write_mp3(request.output_file, combined, sample_rate)
        else:
            _write_wav(request.output_file, combined, sample_rate)
        tts_log.encode_end()

        tts_log.chapter_end(request.output_file.stem)
        tts_log.close()

        del combined
        del model  # drop local ref so unload() can fully free it
        gc.collect()
        torch.cuda.empty_cache()

        return request.output_file

    def _get_base_model(self, *, quantize: bool = False, tts_log: "_TtsLog | None" = None):
        if self._base_model is None or getattr(self, "_base_model_quantize", None) != quantize:
            logger.info("Loading Qwen3-TTS base model from %s (quantize=%s)", self.base_model_id, quantize)
            self._base_model = _load_model(self.base_model_id, self.device, quantize=quantize, tts_log=tts_log)
            self._base_model_quantize = quantize
        return self._base_model

    def unload(self) -> None:
        """Delete all loaded models and release VRAM."""
        import gc
        import torch  # type: ignore

        if torch.cuda.is_available():
            before_alloc = torch.cuda.memory_allocated() // (1024 * 1024)
            before_reserved = torch.cuda.memory_reserved() // (1024 * 1024)
            logger.info("Before unload — allocated: %d MiB, reserved: %d MiB", before_alloc, before_reserved)

        self._base_model = None
        self._custom_models.clear()
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            after_alloc = torch.cuda.memory_allocated() // (1024 * 1024)
            after_reserved = torch.cuda.memory_reserved() // (1024 * 1024)
            logger.info("After unload  — allocated: %d MiB, reserved: %d MiB", after_alloc, after_reserved)

        logger.info("Qwen3TTSProvider unloaded.")

    def _get_custom_model(self, speaker: str, *, quantize: bool = False, tts_log: "_TtsLog | None" = None):
        cache_key = (speaker, quantize)
        if cache_key not in self._custom_models:
            speaker_path = SPEAKERS_DIR / speaker
            if not speaker_path.exists():
                available = [d.name for d in SPEAKERS_DIR.iterdir() if d.is_dir()] if SPEAKERS_DIR.exists() else []
                raise TTSProviderError(
                    f"Custom speaker '{speaker}' not found in {SPEAKERS_DIR}. "
                    f"Available: {available or 'none'}."
                )
            logger.info("Loading Qwen3-TTS custom model for speaker '%s' (quantize=%s)", speaker, quantize)
            self._custom_models[cache_key] = _load_model(str(speaker_path), self.device, quantize=quantize, warmup_speaker=speaker, tts_log=tts_log)
        return self._custom_models[cache_key]
