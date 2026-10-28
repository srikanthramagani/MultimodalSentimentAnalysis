# MIT License
# Multimodal Sentiment Analysis - API Tests

import io
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Patch model loading before importing app so startup does not trigger real downloads
_text_mock_result = {
    "label": "positive",
    "confidence": 0.92,
    "raw_scores": {"positive": 0.92, "neutral": 0.06, "negative": 0.02},
}

_negative_mock_result = {
    "label": "negative",
    "confidence": 0.88,
    "raw_scores": {"positive": 0.05, "neutral": 0.07, "negative": 0.88},
}


_voice_mock_positive = {
    "transcription": "I am having such a fantastic time!",
    "label": "positive",
    "sentiment": "good",
    "confidence": 0.95,
    "raw_scores": {"positive": 0.95, "neutral": 0.03, "negative": 0.02},
}

_voice_mock_negative = {
    "transcription": "This is completely broken and terrible.",
    "label": "negative",
    "sentiment": "bad",
    "confidence": 0.91,
    "raw_scores": {"positive": 0.02, "neutral": 0.07, "negative": 0.91},
}

_voice_mock_neutral = {
    "transcription": "The package arrived at 3 PM today.",
    "label": "neutral",
    "sentiment": "neutral",
    "confidence": 0.84,
    "raw_scores": {"positive": 0.08, "neutral": 0.84, "negative": 0.08},
}


@pytest.fixture(scope="session", autouse=True)
def patch_model_loads():
    with (
        patch("app.models.text_analyzer.load_model"),
        patch("app.models.image_analyzer.load_model"),
        patch("app.models.voice_analyzer.load_model"),
    ):
        yield


@pytest.fixture(scope="session")
def app():
    from app.main import app as fastapi_app
    return fastapi_app


@pytest.fixture
def async_client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_health(async_client):
    async with async_client as client:
        resp = await client.get("/")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_text_only_positive(async_client):
    with patch("app.models.text_analyzer.analyze", return_value=_text_mock_result):
        async with async_client as client:
            resp = await client.post(
                "/api/analyze",
                data={"text": "I absolutely love this, it is wonderful!"},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall"]["label"] == "positive"


@pytest.mark.asyncio
async def test_text_only_negative(async_client):
    with patch("app.models.text_analyzer.analyze", return_value=_negative_mock_result):
        async with async_client as client:
            resp = await client.post(
                "/api/analyze",
                data={"text": "This is absolutely terrible and I hate it."},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall"]["label"] == "negative"


@pytest.mark.asyncio
async def test_invalid_text_too_short(async_client):
    async with async_client as client:
        resp = await client.post("/api/analyze", data={"text": "hi"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_invalid_image_format(async_client):
    fake_file = io.BytesIO(b"this is not an image file at all")
    fake_file.name = "not_an_image.txt"
    with patch("app.models.text_analyzer.analyze", return_value=_text_mock_result):
        async with async_client as client:
            resp = await client.post(
                "/api/analyze",
                data={"text": "Some valid text for analysis here."},
                files={"image": ("not_an_image.txt", fake_file, "image/jpeg")},
            )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_voice_only_positive(async_client):
    fake_audio = io.BytesIO(b"RIFFdummywavdata")
    with patch("app.models.voice_analyzer.analyze", return_value=_voice_mock_positive):
        async with async_client as client:
            resp = await client.post(
                "/api/analyze-voice",
                files={"voice": ("sample.wav", fake_audio, "audio/wav")},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["overall"]["label"] == "positive"
    assert body["overall"]["sentiment"] == "good"
    assert body["voice"]["sentiment"] == "good"
    assert body["voice"]["confidence"] == 0.95


@pytest.mark.asyncio
async def test_voice_only_negative(async_client):
    fake_audio = io.BytesIO(b"RIFFdummywavdata")
    with patch("app.models.voice_analyzer.analyze", return_value=_voice_mock_negative):
        async with async_client as client:
            resp = await client.post(
                "/api/analyze-voice",
                files={"voice": ("sample.wav", fake_audio, "audio/wav")},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall"]["label"] == "negative"
    assert body["overall"]["sentiment"] == "bad"
    assert body["voice"]["sentiment"] == "bad"


@pytest.mark.asyncio
async def test_voice_only_neutral(async_client):
    fake_audio = io.BytesIO(b"RIFFdummywavdata")
    with patch("app.models.voice_analyzer.analyze", return_value=_voice_mock_neutral):
        async with async_client as client:
            resp = await client.post(
                "/api/analyze-voice",
                files={"voice": ("sample.wav", fake_audio, "audio/wav")},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall"]["label"] == "neutral"
    assert body["overall"]["sentiment"] == "neutral"


@pytest.mark.asyncio
async def test_invalid_voice_format(async_client):
    fake_file = io.BytesIO(b"not an audio file")
    async with async_client as client:
        resp = await client.post(
            "/api/analyze-voice",
            files={"voice": ("script.py", fake_file, "text/x-python")},
        )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_multimodal_with_voice(async_client):
    fake_audio = io.BytesIO(b"RIFFdummywavdata")
    with (
        patch("app.models.text_analyzer.analyze", return_value=_text_mock_result),
        patch("app.models.voice_analyzer.analyze", return_value=_voice_mock_positive),
    ):
        async with async_client as client:
            resp = await client.post(
                "/api/analyze",
                data={"text": "Everything was truly wonderful!"},
                files={"voice": ("clip.wav", fake_audio, "audio/wav")},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert "text" in body["modalities_used"]
    assert "voice" in body["modalities_used"]
    assert body["voice"]["sentiment"] == "good"
    assert body["overall"]["label"] == "positive"


@pytest.mark.asyncio
async def test_analyze_with_voice_only(async_client):
    fake_audio = io.BytesIO(b"RIFFdummywavdata")
    with patch("app.models.voice_analyzer.analyze", return_value=_voice_mock_positive):
        async with async_client as client:
            resp = await client.post(
                "/api/analyze",
                files={"voice": ("speech.wav", fake_audio, "audio/wav")},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall"]["label"] == "positive"
    assert body["modalities_used"] == ["voice"]


def test_voice_read_and_resample():
    from app.models import voice_analyzer
    import numpy as np
    import wave

    # Create 0.2s 16kHz sine wave
    sample_rate = 16000
    t = np.linspace(0, 0.2, int(sample_rate * 0.2), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    wav_bytes = buffer.getvalue()

    arr = voice_analyzer._read_and_resample(wav_bytes)
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    assert len(arr) == int(0.2 * 16000)


def test_voice_analyzer_silence_detection():
    from app.models import voice_analyzer
    import numpy as np
    import wave
    from unittest.mock import MagicMock

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 3200)

    voice_analyzer._asr_pipeline = MagicMock()
    res = voice_analyzer.analyze(buffer.getvalue())
    assert res["label"] == "neutral"
    assert res["sentiment"] == "neutral"
    assert res["transcription"] == "(no speech detected)"


