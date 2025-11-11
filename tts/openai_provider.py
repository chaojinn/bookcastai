from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from tts.tts_provider import TTSProvider, TTSProviderError, TTSRequest


class OpenAITTSProvider(TTSProvider):
    """Text-to-speech provider backed by OpenAI's streaming TTS models."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_model: str = "gpt-4o-mini-tts",
        default_voice: str = "alloy",
    ) -> None:
        self.api_key = api_key or self._load_api_key()
        if not self.api_key:
            raise TTSProviderError(
                "OpenAI API key not configured. Set the OPENAI_API_KEY environment variable."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise TTSProviderError(
                "The 'openai' package is required for the OpenAI TTS provider. "
                "Install it with 'pip install openai'."
            ) from exc

        self._client = OpenAI(api_key=self.api_key)
        self.default_model = default_model
        self.default_voice = default_voice

    def tts(self, request: TTSRequest) -> Path:
        params = request.parsed_parameters()

        model = str(params.get("model", self.default_model))
        voice = str(params.get("voice", self.default_voice))
        response_format = str(params.get("format", "mp3"))

        chunks = _split_text_into_chunks(request.text_content, max_chars=1000)
        if not chunks:
            raise TTSProviderError("No text provided for TTS synthesis.")

        tmp_dir = Path(tempfile.mkdtemp(prefix="openai_tts_"))
        chunk_paths: list[Path] = []

        try:
            for index, chunk in enumerate(chunks, start=1):
                chunk_path = tmp_dir / f"chunk_{index:03d}.{response_format}"
                kwargs: dict[str, Any] = {
                    "model": model,
                    "voice": voice,
                    "input": chunk,
                    "response_format": response_format,
                }

                extra_options = params.get("options")
                if isinstance(extra_options, dict):
                    kwargs.update(extra_options)

                try:
                    with self._client.audio.speech.with_streaming_response.create(**kwargs) as response:
                        response.stream_to_file(chunk_path)
                except Exception as exc:  # noqa: BLE001
                    raise TTSProviderError(f"OpenAI TTS request failed: {exc}") from exc

                chunk_paths.append(chunk_path)

            output_path = request.output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _concat_audio(chunk_paths, output_path)
            return output_path
        finally:
            _cleanup_chunks(chunk_paths, tmp_dir)

    def list_voices(self) -> list[str]:
        """Return the available voice identifiers exposed by the API."""
        audio_resource = getattr(self._client, "audio", None)

        if not audio_resource or not hasattr(audio_resource, "voices"):
            # The SDK version may not expose a voices endpoint yet; fall back to known defaults.
            return ["alloy", "verse", "aria", "lumen"]

        try:
            response = audio_resource.voices.list()
        except Exception as exc:  # noqa: BLE001
            raise TTSProviderError(f"Failed to list OpenAI voices: {exc}") from exc

        voices: list[str] = []
        data = getattr(response, "data", None) or getattr(response, "voices", None)

        if isinstance(response, dict):
            data = response.get("data") or response.get("voices")

        if data and isinstance(data, (list, tuple)):
            for item in data:
                if isinstance(item, dict):
                    name = item.get("voice_id") or item.get("name")
                else:
                    name = getattr(item, "voice_id", None) or getattr(item, "name", None)
                if name:
                    voices.append(str(name))

        return voices or ["alloy"]

    @staticmethod
    def _load_api_key() -> str | None:
        key = os.getenv("OPENAI_API_KEY")
        if key:
            return key
        try:
            from dotenv import load_dotenv
        except ImportError:
            return None

        candidate_paths = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parent / ".env",
            Path(__file__).resolve().parent.parent / ".env",
        ]

        for path in candidate_paths:
            if path.is_file():
                load_dotenv(path, override=True)
                key = os.getenv("OPENAI_API_KEY")
                if key:
                    return key

        load_dotenv(override=True)

        return os.getenv("OPENAI_API_KEY")


def _split_text_into_chunks(text: str, *, max_chars: int) -> list[str]:
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_len = len(sentence)
        if current and current_len + sentence_len + 1 > max_chars:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = sentence_len
        else:
            current.append(sentence)
            current_len += sentence_len + 1  # account for space

    if current:
        chunks.append(" ".join(current))

    # Ensure no chunk exceeds max_chars by naive splitting if necessary.
    final_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            for i in range(0, len(chunk), max_chars):
                final_chunks.append(chunk[i : i + max_chars])
    return final_chunks


def _concat_audio(chunks: list[Path], destination: Path) -> None:
    if not chunks:
        raise TTSProviderError("No audio chunks to concatenate.")

    file_list_path = destination.parent / "concat_audio.txt"
    with open(file_list_path, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            resolved = chunk.resolve()
            entry = resolved.as_posix().replace("'", "'\\''")
            handle.write(f"file '{entry}'\n")

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(file_list_path),
        "-c",
        "copy",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise TTSProviderError("ffmpeg executable not found. Install ffmpeg to use OpenAI TTS provider.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        raise TTSProviderError(f"ffmpeg failed to concatenate audio: {stderr}") from exc
    finally:
        file_list_path.unlink(missing_ok=True)


def _cleanup_chunks(chunks: list[Path], directory: Path) -> None:
    for chunk in chunks:
        if chunk.exists():
            chunk.unlink()
    if directory.exists():
        try:
            directory.rmdir()
        except OSError:
            pass
