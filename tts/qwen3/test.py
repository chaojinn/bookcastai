
"""Generate an MP3 from a text file using Qwen3-TTS voice cloning.

Uses the Base model with a reference audio clip (ref.wav) for zero-shot voice
cloning — no fine-tuning required.

Usage:
    python tts/qwen3/test.py <text_file> [options]

Examples:
    python tts/qwen3/test.py chapter1.txt --ref tts/qwen3/data/speaker1/train/ref.wav
    python tts/qwen3/test.py chapter1.txt --ref ref.wav --output chapter1.mp3
    python tts/qwen3/test.py chapter1.txt --ref ref.wav --model Qwen/Qwen3-TTS-12Hz-1.7B-Base
=======
"""

from __future__ import annotations

import argparse

import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import torch

DEFAULT_REF = Path(__file__).parent / "data/speaker1/train/ref.wav"
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path),
             "-codec:a", "libmp3lame", "-qscale:a", "2", str(mp3_path)],
            check=True, capture_output=True,
        )
    except FileNotFoundError:
        print("ffmpeg not found — saving as WAV instead.", file=sys.stderr)
        wav_path.rename(mp3_path.with_suffix(".wav"))
        return
    wav_path.unlink(missing_ok=True)



def main() -> None:
    parser = argparse.ArgumentParser(

        description="Zero-shot voice clone TTS using Qwen3-TTS Base model + ref.wav.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("text_file", type=Path, help="Input .txt file")
    parser.add_argument(
        "--ref", type=Path, default=DEFAULT_REF,
        help="Reference audio clip for voice cloning (3-10 seconds of clean speech)",
    )
    parser.add_argument(
        "--ref_text", default=None,
        help="Transcript of the reference audio (improves cloning quality if provided)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output MP3 path (default: <text_file_stem>.mp3 next to the input)",
    )
    parser.add_argument("--device", default=None, help="cuda:0 / cpu / mps (default: auto)")
    args = parser.parse_args()

    text_file: Path = args.text_file.resolve()
    if not text_file.is_file():
        print(f"Error: {text_file} not found", file=sys.stderr)
        sys.exit(1)

    ref: Path = args.ref.resolve()
    if not ref.is_file():
        print(f"Error: reference audio not found: {ref}", file=sys.stderr)
        sys.exit(1)

    text = text_file.read_text(encoding="utf-8").strip()
    if not text:
        print("Error: input file is empty", file=sys.stderr)
        sys.exit(1)

    output: Path = args.output or text_file.with_suffix(".mp3")
    device = args.device or resolve_device()

    print(f"Model   : {args.model}")
    print(f"Ref     : {ref}")
    print(f"Device  : {device}")
    print(f"Output  : {output}")

    try:
        from qwen_tts import Qwen3TTSModel  # type: ignore
    except ImportError:
        print("qwen-tts not installed. Run: pip install -U qwen-tts", file=sys.stderr)
        sys.exit(1)

    load_kwargs: dict = {"device_map": device, "dtype": torch.bfloat16}
    if torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            load_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

    print("\nLoading model ...")
    model = Qwen3TTSModel.from_pretrained(args.model, **load_kwargs)

    print("Synthesising ...")
    wavs, sr = model.generate_voice_clone(
        text=text,
        ref_audio=str(ref),
        ref_text=args.ref_text,   # None is fine — model will infer it
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)

    sf.write(str(tmp_wav), wavs[0], sr)
    wav_to_mp3(tmp_wav, output)

    print(f"\nSaved: {output}")



if __name__ == "__main__":
    main()
