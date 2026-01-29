from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from collections.abc import Callable

from dotenv import load_dotenv
from tts.kokoro import KokoroTTSProvider
from tts.openai_provider import OpenAITTSProvider
from tts.dia_provider import DiaTTSProvider
from tts.tts_provider import TTSProvider, TTSProviderError, TTSRequest

load_dotenv()


def _build_provider(
    provider_name: str,
    *,
    executable: str,
    voice: str,
    lang: str,
    speed: float,
) -> dict[str, object]:
    provider_name = provider_name.lower()
    if provider_name == "kokoro":
        provider: TTSProvider = KokoroTTSProvider(executable=executable)
        parameters = json.dumps(
            {
                "voice": voice,
                "lang": lang,
                "speed": speed,
                "format": "mp3",
            }
        )
    elif provider_name == "openai":
        provider = OpenAITTSProvider()
        parameters = json.dumps(
            {
                "voice": voice,
                "model": "gpt-4o-mini-tts", #tts-1-hd
                "format": "mp3",
            }
        )
    elif provider_name == "dia":
        provider = DiaTTSProvider()
        parameters = json.dumps(
            {
                "max_new_tokens": 2048,
                "guidance_scale": 3.0,
                "temperature": 1.6,
                "top_p": 0.9,
                "top_k": 45,
            }
        )
    else:
        raise ValueError(f"Unsupported TTS provider: {provider_name}")

    return {
        "name": provider_name,
        "instance": provider,
        "parameters": parameters,
    }


def _slugify(text: str) -> str:
    """Generate a filesystem-friendly slug from the chapter title."""
    text = text.strip()
    if not text:
        return "chapter"
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return cleaned.strip("_") or "chapter"


def _chapter_output_path(base_dir: Path, number: int, title: str) -> Path:
    stem = f"{number:03d}_{_slugify(title)}"
    return base_dir / f"{stem}.mp3"


def _synthesise_chunk(
    provider_info: dict[str, object],
    text: str,
    destination: Path,
) -> None:
    request = TTSRequest(
        text_content=text,
        output_file=destination,
        raw_parameters=provider_info["parameters"],
    )
    provider: TTSProvider = provider_info["instance"]  # type: ignore[assignment]
    provider.tts(request)


def _concat_audio_files(chapter_title: str, parts: list[Path], destination: Path) -> None:
    if not parts:
        raise RuntimeError(f"No audio chunks to concatenate for chapter '{chapter_title}'.")

    manifest_path = destination.with_suffix(".concat.txt")
    try:
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for part in parts:
                escaped = part.resolve().as_posix().replace("'", "\\'")
                manifest.write(f"file '{escaped}'\n")

        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest_path),
            "-c",
            "copy",
            str(destination),
        ]
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:  # noqa: BLE001
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
        raise RuntimeError(
            f"Failed to concatenate audio for chapter '{chapter_title}'. "
            f"ffmpeg exited with {exc.returncode}: {stderr.strip()}"
        ) from exc
    finally:
        if manifest_path.exists():
            manifest_path.unlink()

    for part in parts:
        try:
            part.unlink()
        except FileNotFoundError:
            continue


def convert_epub_to_pod(
    book_data: dict,
    output_dir: Path,
    *,
    voice: str = "af_jessica",
    lang: str,
    speed: float = 0.9,
    overwrite: bool,
    executable: str = "kokoro-tts",
    provider_name: str = "kokoro",
    publish_progress: Callable[[int, str], None] | None = None,
) -> list[Path]:
    """Convert the selected EPUB chapters into MP3 files and return their paths."""
    if not isinstance(book_data, dict):
        raise ValueError("Parsed JSON must be an object with chapter data.")

    chapters = book_data.get("chapters", [])
    if not chapters:
        raise ValueError("No chapters were found in the EPUB file.")

    output_dir.mkdir(parents=True, exist_ok=True)

    provider_info = _build_provider(
        provider_name=provider_name,
        executable=executable,
        voice=voice,
        lang=lang,
        speed=speed,
    )
    #output_base = output_dir / provider_info["name"]
    output_base = output_dir / "audio"
    output_base.mkdir(parents=True, exist_ok=True)

    total_chapters = len(chapters)
    generated_files: list[Path] = []
    for chapter_index, chapter in enumerate(chapters, start=1):
        number = chapter["chapter_number"]
        title = chapter.get("chapter_title") or f"Chapter {number}"
        destination = _chapter_output_path(output_base, number, title)

        if destination.exists() and not overwrite:
            generated_files.append(destination)
            continue

        text = chapter.get("content_text", "").strip()
        if not text:
            continue

        chunks = chapter.get("chunks") or [text]
        chunk_texts = [chunk.strip() for chunk in chunks if isinstance(chunk, str) and chunk.strip()]
        if not chunk_texts:
            continue

        chunk_files: list[Path] = []
        print(f"Processing chapter {number}: {title}")
        for chunk_index, chunk_text in enumerate(chunk_texts, start=1):
            chunk_path = destination.with_name(
                f"{destination.stem}_part_{chunk_index:03d}{destination.suffix}"
            )
            if chunk_path.exists():
                chunk_path.unlink()
            print(
                f"  Chunk {chunk_index}/{len(chunk_texts)}: "
                f"{chapter['chapter_title'] or f'Chapter {number}'}"
            )
            try:
                _synthesise_chunk(provider_info, chunk_text, chunk_path)
            except TTSProviderError as exc:
                raise RuntimeError(
                    f"TTS synthesis failed for chapter '{title}' chunk {chunk_index}: {exc}"
                ) from exc
            chunk_files.append(chunk_path)

        try:
            _concat_audio_files(title, chunk_files, destination)
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to build final audio for chapter '{title}': {exc}") from exc

        print(f"Completed chapter {number}: {destination.name}")
        generated_files.append(destination)
        if publish_progress is not None:
            progress = int((chapter_index / total_chapters) * 100)
            publish_progress(progress, f"processing chapter {chapter_index}/{total_chapters}")

    return generated_files


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert EPUB chapters into MP3 files using a selectable TTS provider.",
    )
    parser.add_argument(
        "book_title",
        help="Book title used to locate {PODS_BASE}/{book_title}/book.json and output directory.",
    )
    parser.add_argument(
        "--voice",
        default="af_heart",
        help="Kokoro voice to use for synthesis (default: af_heart).",
    )
    parser.add_argument(
        "--lang",
        default="en-us",
        help="Language code understood by kokoro-tts (default: en-us).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.9,
        help="Speaking speed multiplier (default: 0.9).",
    )
    parser.add_argument(
        "--overwrite",
        default=True,
        action="store_true",
        help="Overwrite existing MP3 files instead of skipping them.",
    )
    parser.add_argument(
        "--provider",
        choices=["kokoro", "openai", "dia"],
        default="kokoro",
        help="TTS provider to use (kokoro or openai).",
    )
    parser.add_argument(
        "--kokoro-cli",
        default="kokoro-tts",
        help="Name or path of the kokoro-tts executable (default: kokoro-tts).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    pods_base = os.getenv("PODS_BASE", "data")
    base_dir = Path(pods_base).expanduser() / args.book_title
    book_json_path = base_dir / "book.json"

    try:
        book_text = book_json_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        parser.error(f"JSON file not found: {book_json_path}")
        return 2
    except OSError as exc:
        parser.error(f"Failed to read JSON file {book_json_path}: {exc}")
        return 2

    try:
        book_data = json.loads(book_text)
    except json.JSONDecodeError as exc:
        parser.error(f"Failed to parse JSON file {book_json_path}: {exc}")
        return 2

    try:
        generated = convert_epub_to_pod(
            book_data=book_data,
            output_dir=base_dir,
            voice=args.voice,
            lang=args.lang,
            speed=args.speed,
            overwrite=args.overwrite,
            executable=args.kokoro_cli,
            provider_name=args.provider,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
        return 2
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
        return 2

    if generated:
        for path in generated:
            print(path)
    else:
        print("No audio files generated (all selected chapters were empty).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
