"""Utility functions for MGSM-JA (Japanese math word problems) evaluation."""

import re
from typing import Optional


def process_docs(dataset):
    """Filter to Japanese questions only (parquet revision has all languages mixed).

    Uses hiragana/katakana presence to distinguish Japanese from Chinese,
    since both share CJK unified ideographs.
    """
    ja_pattern = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")
    return dataset.filter(lambda doc: bool(ja_pattern.search(doc["question"])))


def _extract_last_number(text: str) -> Optional[str]:
    """Extract the last number from text, handling Japanese and standard formats."""
    # Remove commas from numbers
    text = text.replace(",", "")

    # Find all numbers (integers and decimals, possibly negative)
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]
    return None


def process_results(doc, results):
    """Extract numeric answer from model output and compare to ground truth."""
    prediction = results[0]
    pred_number = _extract_last_number(prediction)

    target = str(doc["answer_number"])
    target_number = _extract_last_number(target)

    if pred_number is not None and target_number is not None:
        try:
            match = float(pred_number) == float(target_number)
        except ValueError:
            match = pred_number == target_number
    else:
        match = False

    return {"exact_match": 1.0 if match else 0.0}
