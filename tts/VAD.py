#!/usr/bin/env python3
"""
VAD.py - Voice Activity Detection using pyannote.audio
Detects silence and noise periods in an audio file (mp3/wav).
"""

import sys
import argparse
from pathlib import Path


def _patch_torchaudio():
    """
    Monkey-patch torchaudio for compatibility with pyannote.audio 3.x on
    torchaudio 2.10+ which removed AudioMetaData, info(), and list_audio_backends().
    """
    import torchaudio
    import soundfile as sf
    from collections import namedtuple

    import torch
    import librosa
    import numpy as np

    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = namedtuple(
            "AudioMetaData",
            ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
        )

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    if not hasattr(torchaudio, "info"):
        def _info(path, backend=None):
            info = sf.info(str(path), format=None if backend is None else None)
            return torchaudio.AudioMetaData(
                sample_rate=info.samplerate,
                num_frames=info.frames,
                num_channels=info.channels,
                bits_per_sample=16,
                encoding="PCM_S",
            )
        torchaudio.info = _info

    # torchaudio 2.10 requires torchcodec for load(); patch with librosa instead.
    _original_torchaudio_load = torchaudio.load
    def _torchaudio_load_compat(uri, frame_offset=0, num_frames=-1, normalize=True,
                                channels_first=True, format=None, buffer_size=4096,
                                backend=None):
        waveform, sr = librosa.load(str(uri), sr=None, mono=False)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]  # (1, T)
        tensor = torch.from_numpy(waveform.copy())
        if frame_offset > 0:
            tensor = tensor[:, frame_offset:]
        if num_frames > 0:
            tensor = tensor[:, :num_frames]
        return tensor, sr
    torchaudio.load = _torchaudio_load_compat


def run_vad(audio_path: str, hf_token: str | None = None):
    import soundfile as sf

    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}")
        sys.exit(1)

    _patch_torchaudio()

    # PyTorch 2.6+ changed torch.load default to weights_only=True, which breaks
    # older pyannote checkpoints that embed pytorch_lightning callback objects.
    # Patch torch.load to force weights_only=False for these trusted HF checkpoints.
    import torch
    _original_torch_load = torch.load
    def _torch_load_compat(*args, **kwargs):
        # lightning_fabric passes weights_only=None; PyTorch 2.6 treats None as True.
        # Force False for trusted pyannote/pytorch_lightning checkpoints.
        if kwargs.get("weights_only") is not False:
            kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _torch_load_compat

    from pyannote.audio import Pipeline

    print("Loading pyannote VAD pipeline...")
    if not hf_token:
        print("Error: HF_TOKEN not found. Set it in .env or pass --token.")
        sys.exit(1)

    pipeline = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection",
        use_auth_token=hf_token,
    )

    if pipeline is None:
        print("Error: failed to load pipeline. Check your HF token and that you have accepted")
        print("the model conditions at https://hf.co/pyannote/voice-activity-detection")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    print(f"Using device: {device}")

    print(f"Running VAD on: {audio_path}\n")
    output = pipeline(str(audio_path))

    # Collect speech segments
    speech_segments = [(seg.start, seg.end) for seg, _, _ in output.itertracks(yield_label=True)]

    # Determine total duration via soundfile (avoids torchaudio.load for mp3 compatibility)
    info = sf.info(str(audio_path))
    total_duration = info.frames / info.samplerate

    # Derive silence/noise segments as gaps between speech
    noise_segments = []
    prev_end = 0.0
    for start, end in speech_segments:
        if start > prev_end + 0.001:
            noise_segments.append((prev_end, start))
        prev_end = end
    if prev_end < total_duration - 0.001:
        noise_segments.append((prev_end, total_duration))

    # Print speech segments
    print(f"=== SPEECH SEGMENTS ({len(speech_segments)}) ===")
    for i, (start, end) in enumerate(speech_segments):
        duration = end - start
        print(f"  [{i+1:3d}] {start:8.3f}s -> {end:8.3f}s  (duration: {duration:.3f}s)")

    print()

    # Print silence/noise segments
    print(f"=== SILENCE / NOISE SEGMENTS ({len(noise_segments)}) ===")
    for i, (start, end) in enumerate(noise_segments):
        duration = end - start
        print(f"  [{i+1:3d}] {start:8.3f}s -> {end:8.3f}s  (duration: {duration:.3f}s)")

    print()

    # Summary
    total_speech = sum(e - s for s, e in speech_segments)
    total_noise = sum(e - s for s, e in noise_segments)
    print(f"=== SUMMARY ===")
    print(f"  Total duration : {total_duration:.3f}s")
    print(f"  Speech         : {total_speech:.3f}s  ({100*total_speech/total_duration:.1f}%)")
    print(f"  Silence/Noise  : {total_noise:.3f}s  ({100*total_noise/total_duration:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Detect speech, silence, and noise in audio using pyannote VAD")
    parser.add_argument("audio", help="Path to audio file (mp3 or wav)")
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace access token (overrides .env / HF_TOKEN env var)",
    )
    args = parser.parse_args()

    import os
    from dotenv import load_dotenv

    # Load .env from project root (two levels up from this file: tts/ -> project root)
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    hf_token = args.token or os.environ.get("HF_TOKEN")

    run_vad(args.audio, hf_token)


if __name__ == "__main__":
    main()
