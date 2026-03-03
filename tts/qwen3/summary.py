'''
input param {speaker_name}
read ./data/{speaker_name}/train/train.jsonl
go through each wav file, get the duration of each file and total words of each text transcription
calculate and print out average words/min of the training set.
'''

import argparse
import glob
import json
import re
import wave
import os
from mutagen.mp3 import MP3
from nltk.corpus import cmudict

_cmu = cmudict.dict()

def _syllables(word):
    clean = re.sub(r"[^a-z']", '', word.lower())
    phones = _cmu.get(clean)
    if phones:
        return sum(1 for p in phones[0] if p[-1].isdigit())
    # fallback: count vowel groups
    return len(re.findall(r'[aeiouy]+', clean)) or 1

def count_syllables(text):
    return sum(_syllables(w) for w in text.split())

def get_wav_duration(path):
    with wave.open(path, 'r') as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / rate  # seconds

def summarize_speaker(speaker_name):
    jsonl_path = os.path.join(os.path.dirname(__file__), 'data', speaker_name, 'train', 'train.jsonl')

    total_duration = 0.0
    total_words = 0

    total_syllables = 0

    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            duration = get_wav_duration(entry['audio'])
            words = len(entry['text'].split())
            total_duration += duration
            total_words += words
            total_syllables += count_syllables(entry['text'])

    minutes = total_duration / 60
    print(f"Total duration: {total_duration:.1f}s ({minutes:.2f} min)")
    print(f"Total words: {total_words}")
    print(f"Average words/min: {total_words / minutes:.1f}")
    print(f"Average syllables/min: {total_syllables / minutes:.1f}")

def summarize_chapter(uid, book_title, chapter):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    book_json_path = os.path.join(project_root, 'pod_data', uid, book_title, 'book.json')

    with open(book_json_path) as f:
        book = json.load(f)

    chapter_num = int(chapter)
    ch = next((c for c in book['chapters'] if c['chapter_number'] == chapter_num), None)
    if ch is None:
        print(f"Chapter {chapter_num} not found in book.json")
        return

    text = ch['content_text']
    total_words = len(text.split())
    total_syllables = count_syllables(text)

    padded = str(chapter_num).zfill(3)
    audio_dir = os.path.join(project_root, 'pod_data', uid, book_title, 'audio', 'qwen3')
    matches = glob.glob(os.path.join(audio_dir, f'{padded}_*.mp3'))
    if not matches:
        print(f"No audio file found matching {padded}_*.mp3 in {audio_dir}")
        return

    audio_path = matches[0]
    duration = MP3(audio_path).info.length  # seconds

    minutes = duration / 60
    print(f"Chapter {chapter_num}: {os.path.basename(audio_path)}")
    print(f"Duration: {duration:.1f}s ({minutes:.2f} min)")
    print(f"Words: {total_words}")
    print(f"Average words/min: {total_words / minutes:.1f}")
    print(f"Average syllables/min: {total_syllables / minutes:.1f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speaker')
    parser.add_argument('--uid')
    parser.add_argument('--book_title')
    parser.add_argument('--chapter')
    args = parser.parse_args()

    if args.speaker:
        summarize_speaker(args.speaker)
    elif args.uid and args.book_title and args.chapter:
        summarize_chapter(args.uid, args.book_title, args.chapter)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
