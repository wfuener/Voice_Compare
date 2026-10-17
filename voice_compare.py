"""
Simple ECAPA-TDNN speaker verification test.

This script compares two audio files and predicts whether they are likely
from the same speaker.

Example usage:

    python voice_compare.py
"""

from pathlib import Path

import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from yt_dlp import YoutubeDL


BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "data"
AUDIO_DIR.mkdir(exist_ok=True)


def main() -> None:
    youtube_links = [
        {
            "url": "https://www.youtube.com/watch?v=lxDNHLgWInU&t=462s",
            "name": "Chris_Williamson",
        },
        {
            "url":  "https://www.youtube.com/watch?v=29yWY63Tvs8",
            "name": "MH370_flight",
        },
    ]

    downloaded_audio_files = []

    for video in youtube_links:
        audio_path = download_youtube_audio(video["url"], video["name"])
        downloaded_audio_files.append(audio_path)

    if len(downloaded_audio_files) >= 2:
        verify_speakers(downloaded_audio_files[0], downloaded_audio_files[1])
    else:
        # You can also compare local files while testing.
        verify_speakers("voice1.wav", "voice2.wav")


def download_youtube_audio(url: str, output_name: str) -> Path:
    """
    Download a YouTube video and extract its audio as a WAV file.

    Args:
        url: The YouTube video URL.
        output_name: The base filename to use for the downloaded audio.

    Returns:
        Path to the extracted WAV audio file.
    """

    output_template = str(AUDIO_DIR / f"{output_name}.%(ext)s")
    wav_path = AUDIO_DIR / f"{output_name}.wav"

    if wav_path.exists():
        print(f"Audio already exists, skipping download: {wav_path}")
        return wav_path

    ydl_options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "quiet": False,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "continuedl": True,
        "socket_timeout": 30,
    }

    print(f"Downloading and extracting audio from: {url}")

    with YoutubeDL(ydl_options) as ydl:
        ydl.download([url])

    if not wav_path.exists():
        raise FileNotFoundError(f"Expected WAV file was not created: {wav_path}")

    return wav_path


def verify_speakers(audio_path_1: str | Path, audio_path_2: str | Path) -> None:
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
    )

    emb1 = get_embedding(audio_path_1, classifier)
    emb2 = get_embedding(audio_path_2, classifier)

    similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
    threshold = 0.6

    print(f"Cosine similarity: {similarity:.4f}")
    print("Same speaker" if similarity > threshold else "Different speakers")


def get_embedding(path: str | Path, classifier: EncoderClassifier) -> torch.Tensor:
    signal, sr = torchaudio.load(str(path))

    # The model expects 16 kHz mono audio.
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)

    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    return classifier.encode_batch(signal).squeeze()


if __name__ == "__main__":
    main()