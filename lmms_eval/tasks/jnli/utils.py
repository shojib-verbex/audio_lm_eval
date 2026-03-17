"""Utility functions for JNLI (Japanese Natural Language Inference) evaluation."""

LABEL_MAP = {
    "entailment": "含意",
    "contradiction": "矛盾",
    "neutral": "中立",
}

LABEL_TO_IDX = {"含意": 0, "矛盾": 1, "中立": 2}


def process_docs(dataset):
    """Map English label strings to Japanese."""

    def _map_doc(doc):
        label_str = doc["label"]
        # Handle both string labels and integer labels
        if isinstance(label_str, int):
            idx_to_label = {0: "entailment", 1: "contradiction", 2: "neutral"}
            label_str = idx_to_label.get(label_str, "neutral")
        return {
            "sentence1": doc["sentence1"],
            "sentence2": doc["sentence2"],
            "label": doc["label"],
            "label_text": LABEL_MAP.get(label_str, label_str),
        }

    return dataset.map(_map_doc)


def process_results(doc, results):
    """Extract and match the predicted label."""
    prediction = results[0].strip()

    # Try exact match first
    if prediction in LABEL_TO_IDX:
        pred_label = prediction
    else:
        # Fuzzy match: check if any label appears in the prediction
        pred_label = None
        for label in LABEL_TO_IDX:
            if label in prediction:
                pred_label = label
                break

    target = doc["label_text"]
    return {"acc": 1.0 if pred_label == target else 0.0}
