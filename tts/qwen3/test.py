"""Generate audio from every fine-tuned checkpoint for a speaker.

Usage:
    python tts/qwen3/test.py <speaker_name> "<text>"

Output:
    tts/qwen3/data/<speaker_name>/c001.wav   (checkpoint-epoch-0)
    tts/qwen3/data/<speaker_name>/c002.wav   (checkpoint-epoch-1)
    ...
"""

import argparse
import gc
import sys
from pathlib import Path

import soundfile as sf  # type: ignore
import torch


_DATA_DIR = Path(__file__).parent / "data"


def main():
    parser = argparse.ArgumentParser(
        description="Test all fine-tuned checkpoints for a speaker."
    )
    parser.add_argument("speaker_name", help="Speaker name used during fine-tuning.")
    parser.add_argument("text", help="Text to synthesize.")
    args = parser.parse_args()

    speaker_dir = _DATA_DIR / args.speaker_name
    if not speaker_dir.exists():
        sys.exit(f"Error: no data directory found for speaker '{args.speaker_name}' at {speaker_dir}")

    checkpoints = sorted(speaker_dir.glob("checkpoint-epoch-*"),
                         key=lambda p: int(p.name.split("-")[-1]))
    if not checkpoints:
        sys.exit(f"Error: no checkpoints found under {speaker_dir}")

    print(f"Found {len(checkpoints)} checkpoint(s) for speaker '{args.speaker_name}'")
    print(f"Text: {args.text!r}\n")

    from qwen_tts import Qwen3TTSModel  # type: ignore

    for idx, ckpt in enumerate(checkpoints, start=1):
        out_path = speaker_dir / f"c{idx:03d}.wav"
        print(f"[{idx}/{len(checkpoints)}] {ckpt.name}  ->  {out_path.name}")

        model = Qwen3TTSModel.from_pretrained(
            str(ckpt),
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        wavs, sr = model.generate_custom_voice(text=args.text, speaker=args.speaker_name)
        sf.write(str(out_path), wavs[0], sr, format="WAV", subtype="PCM_16")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nDone. Audio files saved to {speaker_dir}")


if __name__ == "__main__":
    main()
