# MIT License
# Multimodal Sentiment Analysis - Image Analyzer
#
# Two-stage pipeline: BLIP captioning -> RoBERTa sentiment on the caption.

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from app.models import text_analyzer

_processor = None
_blip_model = None


_BLIP_REVISION = "82a37760796d32b1411fe092ab5d4e227313294b"


def load_model() -> None:
    global _processor, _blip_model
    _processor = BlipProcessor.from_pretrained(  # nosec B615
        "Salesforce/blip-image-captioning-base", revision=_BLIP_REVISION
    )
    _blip_model = BlipForConditionalGeneration.from_pretrained(  # nosec B615
        "Salesforce/blip-image-captioning-base", revision=_BLIP_REVISION
    )
    _blip_model.eval()


def _generate_caption(image: Image.Image) -> str:
    if _processor is None or _blip_model is None:
        raise RuntimeError("Image model not loaded. Call load_model() first.")

    inputs = _processor(images=image, return_tensors="pt")
    output_ids = _blip_model.generate(**inputs, max_new_tokens=50)
    caption = _processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


def analyze(image: Image.Image) -> dict:
    caption = _generate_caption(image)
    sentiment = text_analyzer.analyze(caption)

    return {
        "caption": caption,
        "label": sentiment["label"],
        "confidence": sentiment["confidence"],
        "raw_scores": sentiment["raw_scores"],
    }
