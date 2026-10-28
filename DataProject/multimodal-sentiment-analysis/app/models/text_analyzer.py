# MIT License
# Multimodal Sentiment Analysis - Text Analyzer
#
# Uses cardiffnlp/twitter-roberta-base-sentiment-latest for sentiment classification.

from transformers import pipeline

_classifier = None

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


_ROBERTA_REVISION = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"


def load_model() -> None:
    global _classifier
    _classifier = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        revision=_ROBERTA_REVISION,
        top_k=None,
    )


def analyze(text: str) -> dict:
    if _classifier is None:
        raise RuntimeError("Text model not loaded. Call load_model() first.")

    results = _classifier(text)
    # results is a list of lists when top_k=None: [[{label, score}, ...]]
    scores_raw = results[0] if isinstance(results[0], list) else results

    raw_scores = {}
    for item in scores_raw:
        readable = LABEL_MAP.get(item["label"], item["label"].lower())
        raw_scores[readable] = round(item["score"], 4)

    # Ensure all three keys exist
    for key in ("positive", "neutral", "negative"):
        raw_scores.setdefault(key, 0.0)

    best_label = max(raw_scores, key=raw_scores.__getitem__)
    confidence = raw_scores[best_label]

    return {
        "label": best_label,
        "confidence": confidence,
        "raw_scores": raw_scores,
    }
