# MIT License
# Multimodal Sentiment Analysis - Late Fusion

from typing import Optional

_LABEL_TO_NUMERIC = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


def fuse(
    text_result: Optional[dict] = None,
    image_result: Optional[dict] = None,
    voice_result: Optional[dict] = None,
    text_weight: float = 0.5,
    image_weight: float = 0.25,
    voice_weight: float = 0.25,
) -> dict:
    # Handle single modality cases
    if text_result is not None and image_result is None and voice_result is None:
        return {
            "overall_label": text_result["label"],
            "overall_confidence": text_result["confidence"],
            "text_result": text_result,
            "image_result": None,
            "voice_result": None,
            "fusion_weights": {"text": 1.0, "image": 0.0, "voice": 0.0},
        }

    if voice_result is not None and text_result is None and image_result is None:
        return {
            "overall_label": voice_result["label"],
            "overall_confidence": voice_result["confidence"],
            "text_result": None,
            "image_result": None,
            "voice_result": voice_result,
            "fusion_weights": {"text": 0.0, "image": 0.0, "voice": 1.0},
        }

    # Backward compatibility: text + image only (default 0.6 text / 0.4 image)
    if voice_result is None and text_result is not None and image_result is not None:
        active_weights = {"text": 0.6, "image": 0.4, "voice": 0.0}
    else:
        # Normalize weights among active modalities
        raw_weights = {}
        if text_result is not None:
            raw_weights["text"] = text_weight
        if image_result is not None:
            raw_weights["image"] = image_weight
        if voice_result is not None:
            raw_weights["voice"] = voice_weight

        total_weight = sum(raw_weights.values()) or 1.0
        active_weights = {
            "text": round(raw_weights.get("text", 0.0) / total_weight, 4),
            "image": round(raw_weights.get("image", 0.0) / total_weight, 4),
            "voice": round(raw_weights.get("voice", 0.0) / total_weight, 4),
        }

    # Compute fused numeric score
    fused_numeric = 0.0
    if text_result is not None:
        fused_numeric += active_weights["text"] * _LABEL_TO_NUMERIC[text_result["label"]]
    if image_result is not None:
        fused_numeric += active_weights["image"] * _LABEL_TO_NUMERIC[image_result["label"]]
    if voice_result is not None:
        fused_numeric += active_weights["voice"] * _LABEL_TO_NUMERIC[voice_result["label"]]

    if fused_numeric > 0.2:
        overall_label = "positive"
    elif fused_numeric < -0.2:
        overall_label = "negative"
    else:
        overall_label = "neutral"

    # Weighted average of per-class probabilities
    fused_scores = {}
    for cls in ("positive", "neutral", "negative"):
        score = 0.0
        if text_result is not None:
            score += active_weights["text"] * text_result["raw_scores"].get(cls, 0.0)
        if image_result is not None:
            score += active_weights["image"] * image_result["raw_scores"].get(cls, 0.0)
        if voice_result is not None:
            score += active_weights["voice"] * voice_result["raw_scores"].get(cls, 0.0)
        fused_scores[cls] = round(score, 4)

    overall_confidence = fused_scores[overall_label]

    return {
        "overall_label": overall_label,
        "overall_confidence": overall_confidence,
        "text_result": text_result,
        "image_result": image_result,
        "voice_result": voice_result,
        "fusion_weights": active_weights,
    }
