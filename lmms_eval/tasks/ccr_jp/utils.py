import json
import os
import re
import unicodedata

import numpy as np
import soundfile as sf
from loguru import logger as eval_logger


def load_nemo_manifest(manifest_path):
    """Load a NeMo-format JSONL manifest as a list of dicts."""
    docs = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def ccr_jp_doc_to_audio(doc):
    """Load audio from NeMo manifest entry (audio_filepath field)."""
    audio_path = doc.get("audio_filepath", "")
    if not audio_path or not os.path.exists(audio_path):
        eval_logger.warning(f"Audio file not found: {audio_path}")
        return []

    audio_array, sr = sf.read(audio_path, dtype="float32")

    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    return [{"array": audio_array, "sampling_rate": sr}]


def ccr_jp_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    default_prompt = "この日本語の音声を正確に書き起こしてください。日本語のみで出力し、説明や整形は不要です。"
    return f"{pre_prompt}{default_prompt}{post_prompt}"


def _normalize_japanese(text):
    """Normalize Japanese text for fair CER comparison."""
    # Full-width to half-width for alphanumeric (Ａ→A, １→1)
    text = unicodedata.normalize("NFKC", text)
    # Remove punctuation: Japanese (。、！？・「」『』（）) and ASCII
    text = re.sub(r'[。、！？!?.,;:・「」『』（）()【】\[\]{}""''\'\"~〜…ー\-－—\s]', '', text)
    # Lowercase for any romaji/English
    text = text.lower()
    return text


def ccr_jp_process_results(doc, results):
    """Calculate CER for Japanese transcription."""
    ground_truth = doc.get("text", "").strip()
    if not ground_truth:
        return {"cer": 1.0}

    prediction = results[0].strip() if isinstance(results[0], str) else str(results[0])

    ground_truth = _normalize_japanese(ground_truth)
    prediction = _normalize_japanese(prediction)

    if not ground_truth:
        return {"cer": 0.0 if not prediction else 1.0}

    cer = _calculate_cer(ground_truth, prediction)
    wer = _calculate_wer(ground_truth, prediction)
    return {"cer": cer, "wer": wer}


def _calculate_cer(reference, hypothesis):
    """Character Error Rate for Japanese (character-level edit distance)."""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)

    n, m = len(ref_chars), len(hyp_chars)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1] + 1, dp[i][j - 1] + 1, dp[i - 1][j] + 1)

    return dp[n][m] / max(n, 1)


def _calculate_wer(reference, hypothesis):
    """Word Error Rate for Japanese (split on characters as words are not space-delimited)."""
    # For Japanese, use MeCab/character n-grams or simple character-group splitting
    # Simple approach: split by common particles and treat each as a "word"
    # More practical: use the same edit distance but on word-level tokens
    import MeCab
    tagger = MeCab.Tagger("-Owakati")
    ref_words = tagger.parse(reference).strip().split()
    hyp_words = tagger.parse(hypothesis).strip().split()

    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1] + 1, dp[i][j - 1] + 1, dp[i - 1][j] + 1)

    return dp[n][m] / max(n, 1)


def ccr_jp_cer(results):
    """Aggregate CER scores."""
    if not results:
        return 0.0
    return sum(results) / len(results)


def ccr_jp_wer(results):
    """Aggregate WER scores."""
    if not results:
        return 0.0
    return sum(results) / len(results)
