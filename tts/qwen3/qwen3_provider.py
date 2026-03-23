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
VAD_SILENCE_THRESHOLD = 0.10  # regenerate chunk if silence fraction exceeds this
VAD_MAX_RETRIES = 5           # max regeneration attempts per chunk before using best result

logger = logging.getLogger(__name__)

_QWEN3_MODEL_CLS = None
_SF = None
_AUDIO_SEGMENT = None
_VAD_PIPELINE = None
_TORCHAUDIO_PATCHED = False


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
    """Register a deepcopy handler for dict_keys so bitsandbytes quantization works."""
    import copy
    dict_keys_type = type({}.keys())
    if dict_keys_type not in copy._deepcopy_dispatch:
        copy._deepcopy_dispatch[dict_keys_type] = lambda x, memo: list(x)


def _patch_torchaudio_for_pyannote() -> None:
    """Apply one-time compat patches so pyannote.audio works with torchaudio 2.10+."""
    global _TORCHAUDIO_PATCHED
    if _TORCHAUDIO_PATCHED:
        return

    import torchaudio  # type: ignore
    import torch  # type: ignore
    import numpy as np  # type: ignore
    import librosa  # type: ignore
    import soundfile as sf  # type: ignore
    from collections import namedtuple

    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = namedtuple(
            "AudioMetaData",
            ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
        )

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    if not hasattr(torchaudio, "info"):
        def _info(path, backend=None):
            info = sf.info(str(path))
            return torchaudio.AudioMetaData(
                sample_rate=info.samplerate,
                num_frames=info.frames,
                num_channels=info.channels,
                bits_per_sample=16,
                encoding="PCM_S",
            )
        torchaudio.info = _info

    def _load_compat(uri, frame_offset=0, num_frames=-1, normalize=True,
                     channels_first=True, format=None, buffer_size=4096, backend=None):
        waveform, sr = librosa.load(str(uri), sr=None, mono=False)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        tensor = torch.from_numpy(waveform.copy())
        if frame_offset > 0:
            tensor = tensor[:, frame_offset:]
        if num_frames > 0:
            tensor = tensor[:, :num_frames]
        return tensor, sr
    torchaudio.load = _load_compat

    _orig_torch_load = torch.load
    def _torch_load_compat(*args, **kwargs):
        if kwargs.get("weights_only") is not False:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _torch_load_compat

    _TORCHAUDIO_PATCHED = True


def _load_vad_pipeline():
    """Load and cache the pyannote VAD pipeline (singleton, runs on CPU)."""
    global _VAD_PIPELINE
    if _VAD_PIPELINE is not None:
        return _VAD_PIPELINE

    import os
    import torch  # type: ignore
    from dotenv import load_dotenv  # type: ignore

    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise TTSProviderError("HF_TOKEN not found in .env — required for VAD pipeline")

    _patch_torchaudio_for_pyannote()

    from pyannote.audio import Pipeline  # type: ignore

    pipeline = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection",
        use_auth_token=hf_token,
    )
    if pipeline is None:
        raise TTSProviderError(
            "Failed to load pyannote VAD pipeline. "
            "Check HF_TOKEN and accept terms at https://hf.co/pyannote/voice-activity-detection"
        )

    # Run VAD on CPU to avoid VRAM conflicts with the TTS model on GPU.
    pipeline.to(torch.device("cpu"))
    _VAD_PIPELINE = pipeline
    return _VAD_PIPELINE


def _vad_silence_pct(wav_path: Path, pipeline) -> float:
    """Return the fraction [0, 1] of silence/noise in a wav file using pyannote VAD."""
    import soundfile as sf  # type: ignore

    output = pipeline(str(wav_path))
    speech_segments = [
        (seg.start, seg.end) for seg, _, _ in output.itertracks(yield_label=True)
    ]
    info = sf.info(str(wav_path))
    total_duration = info.frames / info.samplerate
    if total_duration <= 0:
        return 0.0
    total_speech = sum(e - s for s, e in speech_segments)
    return 1.0 - (total_speech / total_duration)


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

    def chunk_start(self, index: int, text: str, attempt: int = 1) -> None:
        single_line = " ".join(text.split())
        self._write_line(f"Chunk {index} attempt {attempt} start: {single_line}")

    def chunk_end(self, index: int) -> None:
        self._write_line(f"Chunk {index} end")

    def chunk_vad_check(self, index: int, attempt: int, copy_idx: int, silence_pct: float) -> None:
        self._write_line(
            f"Chunk {index} attempt {attempt} copy {copy_idx} VAD: silence={silence_pct*100:.1f}%"
        )

    def chunk_vad_requeue(self, index: int, attempt: int, silence_pct: float) -> None:
        self._write_line(
            f"Chunk {index} attempt {attempt} VAD: silence={silence_pct*100:.1f}% > "
            f"{VAD_SILENCE_THRESHOLD*100:.0f}% threshold — requeued (attempt {attempt+1}/{VAD_MAX_RETRIES})"
        )

    def chunk_vad_gave_up(self, index: int, best_silence_pct: float) -> None:
        self._write_line(
            f"Chunk {index} VAD: exhausted {VAD_MAX_RETRIES} retries, "
            f"using best result (silence={best_silence_pct*100:.1f}%)"
        )

    def batch_start(self, batch_num: int, n_jobs: int, n_wavs: int, queue_remaining: int, total: int) -> None:
        copies = f", {n_wavs // n_jobs}x copies to fill batch" if n_wavs > n_jobs else ""
        self._write_line(
            f"Batch {batch_num}: generating {n_jobs} chunks as {n_wavs} wavs{copies} "
            f"({queue_remaining} remaining in queue, {total} total)"
        )

    def batch_end(self, batch_num: int, accepted: int, total: int) -> None:
        self._write_line(f"Batch {batch_num} done: {accepted}/{total} chunks accepted so far")

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

        gen_kwargs: dict = {"speaker": speaker, "max_new_tokens": 1440}
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

        vad_pipeline = _load_vad_pipeline()

        debug_dir = request.output_file.parent / "debug" / request.output_file.stem
        debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving debug chunk WAVs to %s", debug_dir)

        tts_log.chapter_start(request.output_file.stem)

        # Queue entries: dict with keys text, chunk_idx, attempt, best_arr, best_pct
        queue: list[dict] = [
            {"text": c, "chunk_idx": i, "attempt": 1, "best_arr": None, "best_pct": 1.0}
            for i, c in enumerate(chunks)
        ]
        results: dict[int, np.ndarray] = {}
        sample_rate: int | None = None
        batch_num = 0

        while queue:
            batch_jobs = queue[:batch_size]
            queue = queue[batch_size:]
            batch_num += 1
            n_jobs = len(batch_jobs)

            # When fewer jobs than batch_size, distribute spare capacity evenly so
            # the GPU batch is fully utilised and each chunk gets multiple candidates.
            if n_jobs < batch_size:
                base = batch_size // n_jobs
                extra = batch_size % n_jobs
                copies_per_job = [base + (1 if i < extra else 0) for i in range(n_jobs)]
            else:
                copies_per_job = [1] * n_jobs

            expanded_texts = []
            for job, n_copies in zip(batch_jobs, copies_per_job):
                expanded_texts.extend([job["text"]] * n_copies)

            tts_log.batch_start(batch_num, n_jobs, len(expanded_texts), len(queue), len(chunks))
            try:
                with torch.inference_mode():
                    wavs, sr = model.generate_custom_voice(text=expanded_texts, **gen_kwargs)
                if hasattr(wavs, "cpu"):
                    wavs = wavs.cpu()
                elif isinstance(wavs, (list, tuple)):
                    wavs = [w.cpu() if hasattr(w, "cpu") else w for w in wavs]
            except Exception as exc:
                tts_log.close()
                raise TTSProviderError(
                    f"Qwen3-TTS synthesis failed on batch {batch_num}: {exc}"
                ) from exc

            sample_rate = sr
            torch.cuda.empty_cache()

            wav_offset = 0
            for job, n_copies in zip(batch_jobs, copies_per_job):
                chunk_idx = job["chunk_idx"]
                attempt = job["attempt"]
                job_wavs = wavs[wav_offset:wav_offset + n_copies]
                wav_offset += n_copies

                tts_log.chunk_start(chunk_idx, job["text"], attempt)

                best_this_arr, best_this_pct = None, 1.0
                accepted_arr = None

                for copy_idx, wav in enumerate(job_wavs):
                    arr = _wav_to_numpy(wav)
                    arr = _trim_silence(arr, sample_rate)

                    # Write temp, VAD check, rename to embed actual pct
                    tmp_path = debug_dir / f"chunk_{chunk_idx:04d}_{attempt}_{copy_idx}_tmp.wav"
                    _write_wav(tmp_path, arr, sample_rate)
                    silence_pct = _vad_silence_pct(tmp_path, vad_pipeline)
                    pct_int = round(silence_pct * 100)
                    tmp_path.rename(debug_dir / f"chunk_{chunk_idx:04d}_{attempt}_{copy_idx}_{pct_int}.wav")

                    tts_log.chunk_vad_check(chunk_idx, attempt, copy_idx, silence_pct)

                    if silence_pct < best_this_pct:
                        best_this_arr = arr
                        best_this_pct = silence_pct

                    if silence_pct <= VAD_SILENCE_THRESHOLD:
                        accepted_arr = arr
                        break  # good enough — skip remaining copies

                # Promote best from this attempt to overall best if improved
                if best_this_pct < job["best_pct"]:
                    job["best_arr"] = best_this_arr
                    job["best_pct"] = best_this_pct

                if accepted_arr is not None:
                    results[chunk_idx] = accepted_arr
                    logger.debug(
                        "Chunk %d attempt %d accepted (silence=%.1f%%)",
                        chunk_idx, attempt, best_this_pct * 100,
                    )
                elif attempt >= VAD_MAX_RETRIES:
                    tts_log.chunk_vad_gave_up(chunk_idx, job["best_pct"])
                    results[chunk_idx] = job["best_arr"]
                else:
                    tts_log.chunk_vad_requeue(chunk_idx, attempt, job["best_pct"])
                    job["attempt"] += 1
                    queue.append(job)

                tts_log.chunk_end(chunk_idx)

            del wavs
            gc.collect()
            torch.cuda.empty_cache()
            tts_log.batch_end(batch_num, len(results), len(chunks))

            if chunk_progress_callback is not None:
                chunk_progress_callback(len(results), len(chunks))

        if not results:
            tts_log.close()
            raise TTSProviderError("Qwen3-TTS generated no audio output.")

        # Assemble in original chunk order with silence pads between
        tts_log.concat_start(len(chunks))
        all_audio: list[np.ndarray] = []
        for i in range(len(chunks)):
            all_audio.append(results[i])
            if i < len(chunks) - 1:
                all_audio.append(_silence_pad(CHUNK_SILENCE_PAD_MS, sample_rate))
        combined = np.concatenate(all_audio)
        del all_audio, results
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
