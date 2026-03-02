"""Fine-tune Qwen3-TTS-12Hz-1.7B-Base for a custom speaker voice.

Takes the output of tts/audio_processor.py (train.jsonl + WAV files) and runs
the full two-step pipeline:
  1. Extract audio codes  (inlined prepare_data logic)
  2. Fine-tune the model  (delegates to official sft_12hz.py via subprocess)

The official sft_12hz.py + dataset.py are auto-downloaded from GitHub on first run
into tts/qwen3/_scripts/ so you don't need to clone the full repo.

Usage:
    python tts/qwen3/finetune.py <train_jsonl> <speaker_name> [options]

Examples:
    python tts/qwen3/finetune.py tts/qwen3/data/speaker1/train/train.jsonl jane_eyre
    python tts/qwen3/finetune.py .../train.jsonl my_voice --epochs 5 --batch_size 1
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

# Official finetuning scripts are downloaded here on first run
_SCRIPTS_DIR = Path(__file__).parent / "_scripts"
_GITHUB_BASE = "https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/finetuning"
_REQUIRED_SCRIPTS = ["sft_12hz.py", "dataset.py"]


# ---------------------------------------------------------------------------
# Step 0 – download official scripts
# ---------------------------------------------------------------------------

def _download_scripts():
    """Download sft_12hz.py and dataset.py from GitHub if not already present."""
    _SCRIPTS_DIR.mkdir(exist_ok=True)
    for name in _REQUIRED_SCRIPTS:
        dst = _SCRIPTS_DIR / name
        if not dst.exists():
            url = f"{_GITHUB_BASE}/{name}"
            print(f"Downloading {name} …")
            urllib.request.urlretrieve(url, dst)
            print(f"  -> {dst}")
        else:
            print(f"[skip] {name} already downloaded.")


# ---------------------------------------------------------------------------
# Step 1 – extract audio codes (inlined prepare_data.py logic)
# ---------------------------------------------------------------------------

def _extract_audio_codes(
    input_jsonl: Path,
    output_jsonl: Path,
    device: str,
    tokenizer_batch: int,
):
    """Convert WAV files to discrete audio codes and write enriched JSONL.

    Equivalent to the official prepare_data.py but inlined here.
    Uses a smaller default batch to avoid OOM on 16 GB VRAM.
    """
    if output_jsonl.exists():
        print(f"[Step 1] {output_jsonl.name} already exists — skipping code extraction.")
        return

    print(f"[Step 1] Extracting audio codes  device={device}  batch={tokenizer_batch}")
    from qwen_tts import Qwen3TTSTokenizer  # type: ignore

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device_map=device,
    )

    lines = [json.loads(l) for l in open(input_jsonl, encoding="utf-8")]
    n = len(lines)
    final_lines = []

    for start in range(0, n, tokenizer_batch):
        batch = lines[start : start + tokenizer_batch]
        audios = [b["audio"] for b in batch]
        enc = tokenizer.encode(audios)
        for code, line in zip(enc.audio_codes, batch):
            line["audio_codes"] = code.cpu().tolist()
            final_lines.append(line)
        print(f"  encoded {min(start + tokenizer_batch, n)}/{n}")

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for line in final_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"[Step 1] Done → {output_jsonl}")

    # Free GPU memory before the training step loads the 1.7B model
    import torch  # type: ignore
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[Step 1] GPU cache cleared  "
              f"(free: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB)")


# ---------------------------------------------------------------------------
# Step 2 – fine-tune via subprocess
# ---------------------------------------------------------------------------

def _resolve_model_path(model_path: str) -> str:
    """Return a local directory path for the model.

    sft_12hz.py calls shutil.copytree(MODEL_PATH, ...) which requires a real
    local directory.  If `model_path` looks like a HuggingFace repo ID
    (no path separators, not an existing dir), download/resolve it via
    huggingface_hub.snapshot_download so we get the cache path.
    """
    p = Path(model_path)
    if p.exists():
        return str(p.resolve())

    # Looks like a HF repo ID (e.g. "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    print(f"[Step 2] Resolving HuggingFace model path: {model_path}")
    from huggingface_hub import snapshot_download  # type: ignore
    local = snapshot_download(repo_id=model_path)
    print(f"[Step 2] Local model path: {local}")
    return local


def _run_finetune(
    codes_jsonl: Path,
    output_dir: Path,
    speaker_name: str,
    model_path: str,
    batch_size: int,
    epochs: int,
    lr: float,
):
    """Invoke sft_12hz.py in a subprocess.

    Using a subprocess ensures that the tokenizer memory from Step 1 is fully
    released before the 1.7B training model is loaded.
    """
    print(f"\n[Step 2] Fine-tuning  speaker='{speaker_name}'  "
          f"batch={batch_size}  epochs={epochs}  lr={lr}")
    print(f"         (effective batch = {batch_size * 4} via 4 gradient-accum steps)")

    # sft_12hz.py calls shutil.copytree(MODEL_PATH, ...) — must be a local dir
    local_model_path = _resolve_model_path(model_path)

    # sft_12hz.py imports `from dataset import TTSDataset`, so _scripts/ must be on PYTHONPATH
    env = os.environ.copy()
    extra_path = str(_SCRIPTS_DIR)
    env["PYTHONPATH"] = extra_path + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable, str(_SCRIPTS_DIR / "sft_12hz.py"),
        "--init_model_path", local_model_path,
        "--output_model_path", str(output_dir),
        "--train_jsonl", str(codes_jsonl),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--num_epochs", str(epochs),
        "--speaker_name", speaker_name,
    ]

    print("CMD:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-TTS for a custom speaker voice (16 GB VRAM-safe defaults)."
    )
    parser.add_argument("train_jsonl", help="train.jsonl produced by audio_processor.py")
    parser.add_argument("speaker_name", help="Name used at inference: model.generate_custom_voice(speaker=...)")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Training batch size per step. Default 1 for 16 GB VRAM. "
             "Effective batch = batch_size × 4 (gradient accumulation steps hardcoded in sft_12hz.py).",
    )
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to save checkpoints. Defaults to <train_jsonl_parent>/finetune_output/.",
    )
    parser.add_argument(
        "--model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="HuggingFace repo or local path of the base model.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device for audio-code extraction.")
    parser.add_argument(
        "--tokenizer_batch", type=int, default=8,
        help="Batch size for the audio tokenizer in Step 1. Reduce to 4 or 1 if OOM during extraction.",
    )
    args = parser.parse_args()

    train_jsonl = Path(args.train_jsonl).resolve()
    if not train_jsonl.exists():
        sys.exit(f"Error: train.jsonl not found: {train_jsonl}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else train_jsonl.parent / "finetune_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    codes_jsonl = output_dir / "train_with_codes.jsonl"

    # 0. Ensure scripts are present
    _download_scripts()

    # 1. Audio-code extraction
    _extract_audio_codes(train_jsonl, codes_jsonl, args.device, args.tokenizer_batch)

    # 2. Fine-tuning
    _run_finetune(
        codes_jsonl=codes_jsonl,
        output_dir=output_dir,
        speaker_name=args.speaker_name,
        model_path=args.model_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )

    last_epoch = args.epochs - 1
    print(f"\nAll done!  Checkpoints: {output_dir}")
    print(f"\nTest inference:")
    print(f"  from qwen_tts import Qwen3TTSModel")
    print(f"  tts = Qwen3TTSModel.from_pretrained('{output_dir}/checkpoint-epoch-{last_epoch}')")
    print(f"  wavs, sr = tts.generate_custom_voice(text='...', speaker='{args.speaker_name}')")


if __name__ == "__main__":
    main()
