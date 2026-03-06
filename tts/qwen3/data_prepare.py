'''
input:
a folder contains mp3 files {base_folder}/original/
process:
sort mp3 files by name in {base_folder}/original/
run whisper model to generate transcription with sentence level timestamp for mp3 files one by one.
save sentences with duration 6-15 seconds to {base_folder}/train/ as utt00x.wav
pad each wav file with 200ms silence at beggining and end
stop when total utterance clips reach 30 minutes
generate a jsonl file to {base_folder}/train/ as train.jsonl, each line contains the following:
{"audio": {file path}, "text": {transcript}, "ref_audio": "{base_folder}/train/ref.wav"}
'''

import re
import json
import argparse
from pathlib import Path
from pydub import AudioSegment
from faster_whisper import WhisperModel

SENTENCE_END = re.compile(r'[.?!;]+$')


def process_mp3(mp3_path, audio, model, train_folder, silence_200ms,
                ref_audio_path, utt_idx, total_seconds, max_total_seconds, utterances):

    segments, _ = model.transcribe(str(mp3_path), beam_size=5, word_timestamps=True)

    word_buf = []       # words accumulated for current sentence
    pending = None      # (words, start) waiting for next word to determine clip end

    def flush(words, start, end):
        nonlocal utt_idx, total_seconds
        duration = end - start
        if 6.0 <= duration <= 15.0:
            text = "".join(w.word for w in words).strip()
            clip = audio[int(start * 1000):int(end * 1000)]
            padded = silence_200ms + clip + silence_200ms

            utt_name = f"utt{utt_idx:03d}.wav"
            utt_path = train_folder / utt_name
            padded.export(str(utt_path), format="wav")

            utterances.append({
                "audio": str(utt_path),
                "text": text,
                "ref_audio": ref_audio_path,
            })

            total_seconds += padded.duration_seconds
            utt_idx += 1
            print(f"  [{utt_idx:03d}] {utt_name}  {duration:.1f}s  total={total_seconds/60:.1f}min")
            print(f"        \"{text}\"")

    for seg in segments:
        if total_seconds >= max_total_seconds:
            break
        if not seg.words:
            continue

        for word in seg.words:
            if total_seconds >= max_total_seconds:
                break

            # If a sentence is pending, this word's start - 20ms is the clip end
            if pending is not None:
                flush(pending[0], pending[1], word.start - 0.02)
                pending = None

            word_buf.append(word)

            # Check if this word ends a sentence
            if SENTENCE_END.search(word.word.strip()):
                pending = (word_buf, word_buf[0].start)
                word_buf = []

    # Flush any pending sentence at end of file using last word's end as fallback
    if pending is not None:
        flush(pending[0], pending[1], pending[0][-1].end)

    return utt_idx, total_seconds


def main(base_folder: str):
    base_folder = Path(base_folder).resolve()
    original_folder = base_folder / "original"
    train_folder = base_folder / "train"
    train_folder.mkdir(parents=True, exist_ok=True)

    mp3_files = sorted(original_folder.glob("*.mp3"))
    if not mp3_files:
        print(f"No mp3 files found in {original_folder}")
        return

    print(f"Found {len(mp3_files)} mp3 files. Loading whisper model...")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    silence_200ms = AudioSegment.silent(duration=200, frame_rate=24000)

    # Ensure ref.wav exists and is 24kHz
    ref_audio_path = str(train_folder / "ref.wav")
    ref_src = train_folder / "ref.wav"
    if not ref_src.exists():
        print(f"WARNING: {ref_src} not found — place a reference wav there before training.")
    else:
        ref = AudioSegment.from_file(str(ref_src))
        if ref.frame_rate != 24000:
            print(f"Resampling ref.wav from {ref.frame_rate}Hz to 24000Hz...")
            ref.set_frame_rate(24000).export(str(ref_src), format="wav")
    max_total_seconds = 30 * 60

    utterances = []
    total_seconds = 0.0
    utt_idx = 0

    for mp3_file in mp3_files:
        if total_seconds >= max_total_seconds:
            break
        print(f"\nProcessing {mp3_file.name}...")
        audio = AudioSegment.from_mp3(str(mp3_file)).set_frame_rate(24000)
        utt_idx, total_seconds = process_mp3(
            str(mp3_file), audio, model,
            train_folder, silence_200ms, ref_audio_path,
            utt_idx, total_seconds, max_total_seconds, utterances,
        )

    jsonl_path = train_folder / "train.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for utt in utterances:
            f.write(json.dumps(utt, ensure_ascii=False) + "\n")

    print(f"\nDone: {utt_idx} utterances, {total_seconds/60:.1f} min total")
    print(f"JSONL -> {jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare TTS training data from mp3 files")
    parser.add_argument("base_folder", help="Base folder (mp3s in original/, output to train/)")
    args = parser.parse_args()
    main(args.base_folder)
