# Multimodal Sentiment Analysis

![Logo](assets/logo.svg)

A sentiment analysis system that understands text, images, and voice clips together. Uses a multi-stage pipeline: BLIP image captioning converts visual content to natural language, Whisper speech recognition transcribes voice clips, then a fine-tuned RoBERTa model analyzes sentiment from text, image captions, and voice transcriptions. Late fusion combines all signals with configurable weights.

[Demo GIF here]

## What it does

Paste text, optionally upload an image, and/or record/upload a voice clip to receive a combined sentiment verdict (Good/Positive, Neutral, or Bad/Negative). The system generates a natural language caption for images, transcribes voice audio, runs sentiment analysis independently across modalities, and fuses the results.

## Features

- Text sentiment analysis using `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Image understanding via BLIP captioning followed by sentiment classification
- Voice clip sentiment analysis via Whisper (`openai/whisper-tiny`) speech recognition followed by sentiment classification
- Late fusion with dynamic normalization across active modalities (text, image, voice)
- Per-modality confidence scores and probability breakdowns (positive/good, neutral, negative/bad)
- Clean web interface with in-browser microphone recording and audio file upload
- Flexible input modes: text-only, voice-only, image+text, or full multimodal (text + image + voice)

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI, Python 3.11+ |
| Text Sentiment | RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest) |
| Image Understanding | BLIP (Salesforce/blip-image-captioning-base) |
| Voice Understanding | Whisper (openai/whisper-tiny) |
| Audio Processing | soundfile, PyTorch |
| Inference | HuggingFace Transformers, PyTorch |
| Frontend | HTML, CSS, JavaScript |

## Prerequisites

- Python 3.11+
- ~2GB disk space for model cache
- 8GB RAM minimum

## Setup

```bash
git clone https://github.com/SaiVenkataGaneshBandaluppi/multimodal-sentiment-analysis
cd multimodal-sentiment-analysis
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Models (~1.5GB) download automatically on first run and are cached locally in the HuggingFace cache directory.

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8002
```

Open http://localhost:8002

## Usage

1. Enter text in the left panel (3 to 500 characters), or leave blank if providing voice.
2. Optionally drag and drop or click to upload a JPEG or PNG image.
3. Optionally record from your microphone or upload an audio file (WAV, MP3, OGG, FLAC, WebM).
4. Click Analyze.
5. Review the overall sentiment verdict (Good/Positive, Neutral, or Bad/Negative), per-modality breakdowns, captions/transcriptions, and confidence scores.

## API

### 1. Multimodal Analysis
```
POST /api/analyze
Content-Type: multipart/form-data

text:  string (optional if voice is provided, 3-500 chars)
image: file   (optional, JPEG or PNG)
voice: file   (optional, WAV, MP3, OGG, FLAC, WebM)
```

### 2. Standalone Voice Analysis
```
POST /api/analyze-voice
Content-Type: multipart/form-data

voice: file   (required, WAV, MP3, OGG, FLAC, WebM)
```

Example response:

```json
{
  "status": "success",
  "overall": {
    "label": "positive",
    "sentiment": "good",
    "confidence": 0.95
  },
  "voice": {
    "filename": "sample.wav",
    "transcription": "I am having such a fantastic time!",
    "label": "positive",
    "sentiment": "good",
    "confidence": 0.95,
    "scores": { "positive": 0.95, "neutral": 0.03, "negative": 0.02 }
  },
  "modalities_used": ["voice"]
}
```

## Architecture

```
Text Input  ->  RoBERTa Sentiment  -----------------------------> Late Fusion -> Overall Sentiment
                                                                       ^
Image Input -> BLIP Captioning     -> RoBERTa Sentiment ---------------|
                                                                       |
Voice Input -> Whisper ASR (Tiny)  -> RoBERTa Sentiment ---------------|
```

Weights: dynamically normalized across active modalities (configured in `app/models/fusion.py`).


## Author

[SrikanthRamagani](https://github.com/Srikanthramagani)
