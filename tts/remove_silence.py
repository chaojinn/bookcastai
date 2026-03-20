"""Remove long silences from a WAV file.

Usage:
    python tts/remove_silence.py <path/to/file.wav>

Output:
    <path/to/file_fixed.wav>
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).parents[1]))


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python tts/remove_silence.py <path/to/file.wav>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        sys.exit(1)
    if input_path.suffix.lower() != ".wav":
        print(f"Error: input must be a WAV file, got: {input_path.suffix}")
        sys.exit(1)

    import soundfile as sf
    import numpy as np
    from tts.qwen3.qwen3_provider import _compress_long_silences

    print(f"Reading {input_path}...")
    audio, sample_rate = sf.read(str(input_path), dtype="float32", always_2d=False)

    original_duration = len(audio) / sample_rate
    print(f"Original duration: {original_duration:.2f}s  sample_rate: {sample_rate}Hz")

    fixed_audio, compressed = _compress_long_silences(audio, sample_rate)

    if compressed:
        for start_sec, end_sec in compressed:
            print(f"  Compressed silence {start_sec:.2f}s-{end_sec:.2f}s → 2.00s")
    else:
        print("  No long silences found.")

    fixed_duration = len(fixed_audio) / sample_rate
    print(f"Fixed duration:    {fixed_duration:.2f}s  (saved {original_duration - fixed_duration:.2f}s)")

    output_path = input_path.with_stem(input_path.stem + "_fixed")
    sf.write(str(output_path), fixed_audio, sample_rate, format="WAV", subtype="PCM_16")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
