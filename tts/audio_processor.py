'''
input:
a folder contains mp3 files {base_folder}
process:
use whisper model to generate transcription with sentence level timestamp for mp3 files one by one.
split the audio into utterance around 8~10 seconds using sentence as boundary.
ignore any utterance >15 seconds
all utterance files (.wav) need to be stored under {base_folder}/train
a jsonl file need to be generated using following format
{"audio": "{base_folder}/train/utt0001.wav", "text": "The transcript of this clip.", "ref_audio": ".{base_folder}/train/ref.wav"}
{"audio": "{base_folder}/train/utt0002.wav", "text": "Another sentence spoken by the speaker.", "ref_audio": "{base_folder}/train/ref.wav"}
don't worry about ref.wav. I will create it manually later.
stop when total utterance clips reach 30 minutes
'''

import os
import sys
import json
import warnings
import numpy as np
import soundfile as sf
from pathlib import Path
from pydub import AudioSegment


TARGET_SECONDS = 30 * 60   # stop after 30 minutes of utterances
MIN_UTT = 8.0              # minimum utterance duration (seconds)
MAX_UTT = 15.0             # skip utterances longer than this
IDEAL_MAX = 10.0           # try not to exceed this before cutting
SAMPLE_RATE = 24000        # output WAV sample rate (Qwen3-TTS expects 24 kHz)


def _load_whisperx():
    warnings.filterwarnings("ignore", message=".*TF32.*", category=UserWarning)
    os.environ.setdefault("ORT_DISABLE_ALL_DEVICE_DISCOVERY", "1")

    import torch
    import torchaudio

    # Compatibility patches for pyannote-audio with torchaudio >= 2.1
    if not hasattr(torchaudio, 'AudioMetaData'):
        torchaudio.AudioMetaData = type('AudioMetaData', (), {})
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: ['soundfile', 'sox']

    # PyTorch 2.6+ defaults weights_only=True; pyannote checkpoints need False
    _orig_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load

    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[whisperx] device={device}, compute_type={compute_type}")

    model = whisperx.load_model("base", device, compute_type=compute_type)
    return model, device


def _transcribe(model, device, audio_path: str):
    import whisperx

    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)
    segments = result.get("segments", [])
    if not segments:
        return []

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device,
    )
    aligned = whisperx.align(
        segments, model_a, metadata, audio, device,
        return_char_alignments=False,
    )
    # Use sentence-level segments (not word_segments)
    return aligned.get("segments", segments)


def _save_utterance(pydub_audio, group, utt_index, train_dir, base_folder, jsonl_lines):
    """Slice audio, resample to SAMPLE_RATE, save WAV, append JSONL line."""
    start_ms = int(group[0]["start"] * 1000)
    end_ms = int(group[-1]["end"] * 1000)
    text = " ".join(s["text"].strip() for s in group)
    duration = (end_ms - start_ms) / 1000.0

    utt_name = f"utt{utt_index:04d}.wav"
    utt_path = train_dir / utt_name

    clip = pydub_audio[start_ms:end_ms]
    # Resample to SAMPLE_RATE, mono
    clip = clip.set_frame_rate(SAMPLE_RATE).set_channels(1)
    samples = np.array(clip.get_array_of_samples(), dtype=np.float32)
    samples /= float(2 ** (clip.sample_width * 8 - 1))  # normalise to [-1, 1]

    sf.write(str(utt_path), samples, SAMPLE_RATE)

    ref_path = str(Path(base_folder) / "train" / "ref.wav")
    jsonl_lines.append({
        "audio": str(utt_path),
        "text": text,
        "ref_audio": ref_path,
    })

    preview = text[:60] + ("…" if len(text) > 60 else "")
    print(f'  saved {utt_name}  {duration:.1f}s  "{preview}"')
    return duration


def process(base_folder: str):
    base_path = Path(base_folder).resolve()
    train_dir = base_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    mp3_files = sorted(base_path.glob("*.mp3"))
    if not mp3_files:
        print("No MP3 files found in", base_folder)
        return

    print(f"Found {len(mp3_files)} MP3 file(s). Loading WhisperX…")
    model, device = _load_whisperx()

    utt_index = 1
    total_seconds = 0.0
    jsonl_lines = []

    for mp3_path in mp3_files:
        if total_seconds >= TARGET_SECONDS:
            break

        print(f"\n[{mp3_path.name}] Transcribing…")
        pydub_audio = AudioSegment.from_mp3(str(mp3_path))
        segments = _transcribe(model, device, str(mp3_path))

        if not segments:
            print("  No segments found, skipping.")
            continue

        # Group segments into utterances of ~8-10 seconds
        current_group = []
        current_dur = 0.0

        def flush(group, dur):
            nonlocal utt_index, total_seconds
            if not group or dur < 1.0:
                return
            if dur > MAX_UTT:
                print(f"  skipped group ({dur:.1f}s > {MAX_UTT}s)")
                return
            saved = _save_utterance(pydub_audio, group, utt_index, train_dir, base_folder, jsonl_lines)
            utt_index += 1
            total_seconds += saved

        for seg in segments:
            if total_seconds >= TARGET_SECONDS:
                break

            start = seg.get("start") or 0.0
            end = seg.get("end") or 0.0
            seg_dur = end - start

            if seg_dur <= 0:
                continue

            if seg_dur > MAX_UTT:
                # Single segment too long — flush whatever we have, skip this
                flush(current_group, current_dur)
                current_group = []
                current_dur = 0.0
                print(f"  skipped segment ({seg_dur:.1f}s > {MAX_UTT}s)")
                continue

            if current_dur + seg_dur > IDEAL_MAX and current_dur >= MIN_UTT:
                # We have enough — save current group, start fresh with this segment
                flush(current_group, current_dur)
                current_group = [seg]
                current_dur = seg_dur
            else:
                current_group.append(seg)
                current_dur += seg_dur

        # Flush the last group for this file
        if total_seconds < TARGET_SECONDS:
            flush(current_group, current_dur)

    # Write JSONL
    jsonl_path = train_dir / "train.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for line in jsonl_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"\nDone. {utt_index - 1} utterances, {total_seconds / 60:.1f} minutes.")
    print(f"JSONL written to: {jsonl_path}")
    print(f"Remember to add ref.wav at: {train_dir / 'ref.wav'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio_processor.py <base_folder>")
        sys.exit(1)
    process(sys.argv[1])
