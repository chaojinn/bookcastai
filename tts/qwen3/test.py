"""Test Qwen3TTSProvider end-to-end.

Synthesises the supplied text and writes audio to the project-root test/ folder.
HF_TOKEN is loaded from the project root .env file automatically.

Usage:
    python tts/qwen3/test.py "Hello world, this is a test."
    python tts/qwen3/test.py "Hello." --speaker Vivian --format mp3
    python tts/qwen3/test.py "Hello." --speaker jane_eyre --output custom.wav
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root and add it to sys.path so tts.* imports work when
# this script is run directly (e.g. python tts/qwen3/test.py ...).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEST_OUTPUT_DIR = PROJECT_ROOT / "test"


def _load_env(env_path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ."""
    if not env_path.exists():
        return
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TTS audio using Qwen3TTSProvider."
    )
    parser.add_argument("text", help="Text to synthesise")
    parser.add_argument(
        "--speaker",
        default="Serena",
        help="Speaker name. Internal: Serena, Vivian, Aiden, Ryan. "
             "Custom: any folder name under tts/qwen3/speakers/. (default: Serena)",
    )
    parser.add_argument(
        "--format",
        choices=["wav", "mp3"],
        default="wav",
        help="Output audio format (default: wav)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (optional)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device (default: cuda:0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output filename (without directory). Saved under test/. "
             "Defaults to qwen3_<speaker>.<format>",
    )
    args = parser.parse_args()

    # Load .env so HF_TOKEN is available before any HuggingFace downloads.
    _load_env(PROJECT_ROOT / ".env")

    TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    output_name = args.output or f"qwen3_{args.speaker}.{args.format}"
    output_file = TEST_OUTPUT_DIR / output_name

    import json
    from tts.qwen3.qwen3_provider import Qwen3TTSProvider
    from tts.tts_provider import TTSRequest

    params: dict = {"speaker": args.speaker, "format": args.format}
    if args.temperature is not None:
        params["temperature"] = args.temperature

    request = TTSRequest(
        text_content=args.text,
        output_file=output_file,
        raw_parameters=json.dumps(params),
    )

    provider = Qwen3TTSProvider(device=args.device)

    print(f"Speaker : {args.speaker}")
    print(f"Format  : {args.format}")
    print(f"Output  : {output_file}")
    print(f"Text    : {args.text[:80]}{'…' if len(args.text) > 80 else ''}")
    print()

    result = provider.tts(request)
    print(f"\nDone. Audio saved to: {result}")


if __name__ == "__main__":
    main()
