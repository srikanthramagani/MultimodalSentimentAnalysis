# MIT License
# Multimodal Sentiment Analysis - Sentiment Router

import io
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.models import fusion, image_analyzer, text_analyzer, voice_analyzer

logger = logging.getLogger(__name__)
router = APIRouter()

_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}
_ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm", ".aac"}
_ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/x-m4a",
    "audio/m4a",
    "audio/flac",
    "audio/webm",
    "application/octet-stream",
}
_MAX_TEXT_LEN = 500
_MIN_TEXT_LEN = 3


def _validate_audio_file(file: UploadFile) -> None:
    filename = (file.filename or "").lower()
    has_valid_ext = any(filename.endswith(ext) for ext in _ALLOWED_AUDIO_EXTENSIONS)
    has_valid_ct = (file.content_type or "").lower() in _ALLOWED_AUDIO_TYPES or (
        file.content_type or ""
    ).startswith("audio/")
    if not (has_valid_ext or has_valid_ct):
        raise HTTPException(
            status_code=422,
            detail="Voice file must be a valid audio format (WAV, MP3, OGG, FLAC, M4A, WEBM).",
        )


@router.post("/api/analyze-voice")
@router.post("/api/analyze/voice")
async def analyze_voice(
    voice: UploadFile = File(...),
):
    """Analyzes a standalone voice clip to determine if sentiment is good, bad, or neutral."""
    if not voice.filename:
        raise HTTPException(status_code=422, detail="No voice file provided.")

    _validate_audio_file(voice)

    try:
        raw_bytes = await voice.read()
        if len(raw_bytes) == 0:
            raise HTTPException(status_code=422, detail="The uploaded voice file is empty.")
        result = voice_analyzer.analyze(raw_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Voice analysis failed: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=f"Unable to process audio file: {exc}",
        )

    return {
        "status": "success",
        "overall": {
            "label": result["label"],
            "sentiment": result["sentiment"],
            "confidence": round(result["confidence"], 4),
        },
        "voice": {
            "filename": voice.filename,
            "transcription": result["transcription"],
            "label": result["label"],
            "sentiment": result["sentiment"],
            "confidence": round(result["confidence"], 4),
            "scores": result["raw_scores"],
        },
        "modalities_used": ["voice"],
    }


@router.post("/api/analyze")
async def analyze_sentiment(
    text: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
    voice: Optional[UploadFile] = File(default=None),
):
    # Ensure at least one modality of text or voice is provided
    has_text = text is not None and text.strip() != ""
    has_voice = voice is not None and bool(voice.filename)

    if not has_text and not has_voice:
        raise HTTPException(
            status_code=422,
            detail="Text or voice clip must be provided for analysis.",
        )

    # Validate text length when text is provided
    text_result = None
    if has_text:
        if len(text) < _MIN_TEXT_LEN:
            raise HTTPException(
                status_code=422,
                detail=f"Text must be at least {_MIN_TEXT_LEN} characters.",
            )
        if len(text) > _MAX_TEXT_LEN:
            raise HTTPException(
                status_code=422,
                detail=f"Text must be at most {_MAX_TEXT_LEN} characters.",
            )

        safe_text_log = text.replace("\r", "").replace("\n", " ")[:80]
        logger.info("Analyze request received. text_preview=%r", safe_text_log)
        text_result = text_analyzer.analyze(text)

    # Validate and process image
    pil_image_result = None
    if image is not None and image.filename:
        if image.content_type not in _ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=422,
                detail="Image must be JPEG or PNG.",
            )

        raw_bytes = await image.read()
        try:
            pil_img = Image.open(io.BytesIO(raw_bytes))
            pil_img.verify()
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except (UnidentifiedImageError, Exception) as exc:
            logger.warning("Invalid image upload rejected: %s", type(exc).__name__)
            raise HTTPException(
                status_code=422,
                detail="The uploaded file is not a valid image.",
            )

        pil_image_result = image_analyzer.analyze(pil_img)

    # Validate and process voice
    voice_result = None
    if has_voice:
        _validate_audio_file(voice)
        try:
            audio_bytes = await voice.read()
            if len(audio_bytes) == 0:
                raise HTTPException(status_code=422, detail="The uploaded voice file is empty.")
            voice_result = voice_analyzer.analyze(audio_bytes)
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Voice processing failed: %s", exc)
            raise HTTPException(
                status_code=422,
                detail=f"Unable to process audio file: {exc}",
            )

    fused = fusion.fuse(
        text_result=text_result,
        image_result=pil_image_result,
        voice_result=voice_result,
    )

    modalities = []
    if text_result is not None:
        modalities.append("text")
    if pil_image_result is not None:
        modalities.append("image")
    if voice_result is not None:
        modalities.append("voice")

    image_payload = None
    if pil_image_result is not None:
        image_payload = {
            "caption": pil_image_result["caption"],
            "label": pil_image_result["label"],
            "confidence": round(pil_image_result["confidence"], 4),
            "scores": pil_image_result["raw_scores"],
        }

    voice_payload = None
    if voice_result is not None:
        voice_payload = {
            "filename": voice.filename,
            "transcription": voice_result["transcription"],
            "label": voice_result["label"],
            "sentiment": voice_result["sentiment"],
            "confidence": round(voice_result["confidence"], 4),
            "scores": voice_result["raw_scores"],
        }

    text_payload = None
    if text_result is not None:
        text_payload = {
            "input": text,
            "label": text_result["label"],
            "confidence": round(text_result["confidence"], 4),
            "scores": text_result["raw_scores"],
        }

    return {
        "overall": {
            "label": fused["overall_label"],
            "confidence": round(fused["overall_confidence"], 4),
        },
        "text": text_payload,
        "image": image_payload,
        "voice": voice_payload,
        "fusion_weights": fused["fusion_weights"],
        "modalities_used": modalities,
    }
