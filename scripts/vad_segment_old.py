import torch
import torchaudio
from silero_vad import load_silero_vad, get_speech_timestamps


def load_audio(path):
    wav, sr = torchaudio.load(path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        sr = 16000

    return wav.squeeze(0), sr


def split_by_vad(audio_path):
    model = load_silero_vad()

    wav, sr = load_audio(audio_path)

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sr,
        threshold=0.5
    )

    segments = []

    for t in speech_timestamps:
        start = t["start"] / sr
        end = t["end"] / sr

        duration = end - start

        # filter cực quan trọng cho voice clone
        if 1.0 <= duration <= 15.0:
            segments.append((start, end))

    return segments