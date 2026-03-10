'''
input 2 audio files (wav/mp3)
process:
if any of the 2 files duration < 10 seconds, report error.
if audio file duration > 30 seconds, get first 30 seconds of audio.
convert both clips to 16kHz mono wav format
use SpeechBrain score, _ = verifier.verify_files("voice1.wav", "voice2.wav") to get similarity score
print out the score.
add any new dependency to requirements.txt
'''

import sys
import tempfile
import os
from pydub import AudioSegment

# Patch torchaudio before speechbrain imports to avoid missing torchcodec/list_audio_backends
import soundfile as sf
import torch
import torchaudio

if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']

def _sf_load(path, channels_first=True, **kwargs):
    data, sr = sf.read(str(path), always_2d=True)  # (samples, channels)
    tensor = torch.from_numpy(data.T if channels_first else data).float()
    return tensor, sr

torchaudio.load = _sf_load

from speechbrain.inference.speaker import SpeakerRecognition


def load_and_prepare(path: str, tmp_dir: str, label: str) -> str:
    audio = AudioSegment.from_file(path)
    duration_sec = len(audio) / 1000.0

    if duration_sec < 10:
        print(f"Error: {label} ({path}) is {duration_sec:.1f}s — must be at least 10 seconds.")
        sys.exit(1)

    if duration_sec > 30:
        audio = audio[:30_000]

    audio = audio.set_frame_rate(16000).set_channels(1)

    out_path = os.path.join(tmp_dir, f"{label}.wav")
    audio.export(out_path, format="wav")
    return out_path


def main():
    if len(sys.argv) != 3:
        print("Usage: python audio_compare.py <audio1> <audio2>")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav1 = load_and_prepare(file1, tmp_dir, "voice1")
        wav2 = load_and_prepare(file2, tmp_dir, "voice2")

        model_dir = os.path.join(tmp_dir, "model")
        verifier = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=model_dir,
        )

        score, _ = verifier.verify_files(wav1, wav2)
        print(f"Similarity score: {score.item():.4f}")


if __name__ == "__main__":
    main()
