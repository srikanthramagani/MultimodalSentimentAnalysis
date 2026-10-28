# MIT License
# Multimodal Sentiment Analysis - Voice Analyzer
#
# Two-stage pipeline: Whisper ASR transcribes voice clip -> RoBERTa sentiment on transcription.

import io
import logging
import wave
from typing import Union

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import pipeline

from app.models import text_analyzer

logger = logging.getLogger(__name__)

_asr_pipeline = None

SENTIMENT_MAP = {
    "positive": "good",
    "neutral": "neutral",
    "negative": "bad",
}


def load_model() -> None:
    global _asr_pipeline
    logger.info("Initializing Whisper ASR pipeline (openai/whisper-tiny)...")
    _asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        device="cpu",
    )
    logger.info("Whisper ASR model loaded successfully.")


def _read_and_resample(audio_bytes: bytes) -> np.ndarray:
    """Reads audio bytes and resamples to 16kHz mono float32 numpy array."""
    data = None
    sr = None

    # Try reading with soundfile first (supports WAV, MP3, OGG, FLAC)
    try:
        buffer = io.BytesIO(audio_bytes)
        data, sr = sf.read(buffer, dtype="float32")
    except Exception as sf_err:
        logger.debug("soundfile failed to read audio (%s), attempting wave fallback", sf_err)
        # Fallback to standard library wave module for PCM WAV
        try:
            buffer = io.BytesIO(audio_bytes)
            with wave.open(buffer, "rb") as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)

                if sampwidth == 2:
                    raw_arr = np.frombuffer(frames, dtype=np.int16)
                    data = raw_arr.astype(np.float32) / 32768.0
                elif sampwidth == 1:
                    raw_arr = np.frombuffer(frames, dtype=np.uint8)
                    data = (raw_arr.astype(np.float32) - 128.0) / 128.0
                elif sampwidth == 4:
                    raw_arr = np.frombuffer(frames, dtype=np.int32)
                    data = raw_arr.astype(np.float32) / 2147483648.0
                else:
                    raise ValueError(f"Unsupported sample width: {sampwidth}")

                if n_channels > 1:
                    data = data.reshape(-1, n_channels)
        except Exception as wave_err:
            raise ValueError(f"Could not decode audio data: {wave_err}") from wave_err

    if data is None or len(data) == 0:
        raise ValueError("Audio data is empty.")

    # Convert multi-channel to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample to 16,000 Hz if necessary
    target_sr = 16000
    if sr != target_sr and len(data) > 0:
        tensor_audio = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
        target_len = max(1, int(round(len(data) * target_sr / sr)))
        resampled = F.interpolate(
            tensor_audio, size=target_len, mode="linear", align_corners=False
        ).squeeze()
        data = resampled.numpy()

    return data.astype(np.float32)


def analyze(audio_input: Union[bytes, io.BytesIO]) -> dict:
    """Analyzes a voice clip to recognize whether sentiment is good, bad, or neutral."""
    if _asr_pipeline is None:
        raise RuntimeError("Voice model not loaded. Call load_model() first.")

    if isinstance(audio_input, io.BytesIO):
        audio_bytes = audio_input.getvalue()
    elif isinstance(audio_input, bytes):
        audio_bytes = audio_input
    else:
        raise TypeError(f"Expected bytes or BytesIO, got {type(audio_input)}")

    audio_arr = _read_and_resample(audio_bytes)

    # Check for empty or silent clip
    if len(audio_arr) == 0 or np.max(np.abs(audio_arr)) < 1e-4:
        return {
            "transcription": "(no speech detected)",
            "label": "neutral",
            "sentiment": "neutral",
            "confidence": 1.0,
            "raw_scores": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
        }

    # Transcribe speech with Whisper
    asr_result = _asr_pipeline(
        {"array": audio_arr, "sampling_rate": 16000},
        generate_kwargs={"task": "transcribe"},
    )

    transcription = asr_result.get("text", "").strip()

    # If transcription is empty or just punctuation
    cleaned = "".join(ch for ch in transcription if ch.isalnum() or ch.isspace()).strip()
    if not cleaned:
        return {
            "transcription": "(no speech detected)",
            "label": "neutral",
            "sentiment": "neutral",
            "confidence": 1.0,
            "raw_scores": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
        }

    # Analyze sentiment of the transcribed speech
    sentiment = text_analyzer.analyze(transcription)
    label = sentiment["label"]

    return {
        "transcription": transcription,
        "label": label,
        "sentiment": SENTIMENT_MAP.get(label, label),
        "confidence": sentiment["confidence"],
        "raw_scores": sentiment["raw_scores"],
    }
