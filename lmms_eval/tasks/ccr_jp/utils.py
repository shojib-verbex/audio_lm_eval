import json
import os
import re
import unicodedata

import editdistance
import numpy as np
import soundfile as sf
from loguru import logger as eval_logger

# Cached singletons to avoid repeated instantiation (which leaks memory)
_fugashi_tagger = None
_pykakasi_instance = None


def _get_fugashi_tagger():
    global _fugashi_tagger
    if _fugashi_tagger is None:
        import fugashi
        _fugashi_tagger = fugashi.Tagger()
    return _fugashi_tagger


def _get_pykakasi():
    global _pykakasi_instance
    if _pykakasi_instance is None:
        import pykakasi
        _pykakasi_instance = pykakasi.kakasi()
    return _pykakasi_instance


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
    default_prompt = "この日本語の音声を正確に書き起こしてください。漢字は一切使わず、すべてひらがなのみで出力してください。説明や整形は不要です。"
    return f"{pre_prompt}{default_prompt}{post_prompt}"


def _digits_to_japanese(text):
    """Convert digit sequences to Japanese hiragana number words."""
    digit_map = {
        '0': 'ぜろ', '1': 'いち', '2': 'に', '3': 'さん', '4': 'よん',
        '5': 'ご', '6': 'ろく', '7': 'なな', '8': 'はち', '9': 'きゅう',
    }

    def _int_to_japanese(n):
        if n == 0:
            return 'ぜろ'
        parts = []
        if n >= 100000000:
            q, n = divmod(n, 100000000)
            parts.append((_int_to_japanese(q) if q > 1 else '') + 'おく')
        if n >= 10000:
            q, n = divmod(n, 10000)
            parts.append((_int_to_japanese(q) if q > 1 else '') + 'まん')
        if n >= 1000:
            q, n = divmod(n, 1000)
            prefix = '' if q == 1 else _int_to_japanese(q)
            if q == 3:
                parts.append('さんぜん')
            elif q == 8:
                parts.append('はっせん')
            else:
                parts.append(prefix + 'せん')
        if n >= 100:
            q, n = divmod(n, 100)
            if q == 3:
                parts.append('さんびゃく')
            elif q == 6:
                parts.append('ろっぴゃく')
            elif q == 8:
                parts.append('はっぴゃく')
            else:
                prefix = '' if q == 1 else _int_to_japanese(q)
                parts.append(prefix + 'ひゃく')
        if n >= 10:
            q, n = divmod(n, 10)
            prefix = '' if q == 1 else _int_to_japanese(q)
            parts.append(prefix + 'じゅう')
        if n > 0:
            parts.append(digit_map[str(n)])
        return ''.join(parts)

    def _replace_number(m):
        num_str = m.group(0)
        try:
            n = int(num_str)
            if n <= 999999999:
                return _int_to_japanese(n)
        except ValueError:
            pass
        return ''.join(digit_map.get(c, c) for c in num_str)

    return re.sub(r'\d+', _replace_number, text)


def _kata_to_hira(text):
    """Convert katakana characters to hiragana via Unicode offset."""
    result = []
    for ch in text:
        cp = ord(ch)
        # Katakana range: 0x30A1-0x30F6 → Hiragana: 0x3041-0x3096
        if 0x30A1 <= cp <= 0x30F6:
            result.append(chr(cp - 0x60))
        else:
            result.append(ch)
    return ''.join(result)


def _to_hiragana(text):
    """Convert kanji and katakana to hiragana using fugashi (MeCab wrapper).

    Fugashi provides clean named-attribute access to MeCab features including
    readings. Falls back to pykakasi for any tokens without a reading.
    """
    try:
        tagger = _get_fugashi_tagger()
    except (ImportError, RuntimeError):
        # fugashi not available, fall back to pykakasi entirely
        kakasi = _get_pykakasi()
        return ''.join(item['hira'] for item in kakasi.convert(text))

    words = tagger(text)
    parts = []
    fallback_surfaces = []
    for word in words:
        # fugashi with UniDic: word.feature has named attributes like .kana, .pron
        # fugashi with IPAdic: word.feature is a namedtuple with positional access
        reading = None

        # Try UniDic named attributes first
        for attr in ('kana', 'pron'):
            val = getattr(word.feature, attr, None)
            if val and val != '*':
                reading = val
                break

        # Try positional access (IPAdic: reading at index 7, pronunciation at 8)
        if not reading:
            try:
                for idx in (7, 8):
                    val = word.feature[idx]
                    if val and val != '*':
                        reading = val
                        break
            except (IndexError, TypeError):
                pass

        if reading:
            parts.append(_kata_to_hira(reading))
        elif all(0x3040 <= ord(c) <= 0x309F for c in word.surface):
            # Already hiragana
            parts.append(word.surface)
        else:
            # Collect for pykakasi fallback
            parts.append(None)
            fallback_surfaces.append((len(parts) - 1, word.surface))

    # Use pykakasi as fallback for tokens without readings
    if fallback_surfaces:
        kakasi = _get_pykakasi()
        for idx, surface in fallback_surfaces:
            result = kakasi.convert(surface)
            parts[idx] = ''.join(item['hira'] for item in result)

    return ''.join(p for p in parts if p is not None)


def _normalize_japanese(text):
    """Normalize Japanese text for fair CER/WER comparison.

    Converts all text to hiragana so that kanji/katakana/hiragana
    differences don't affect the error rate.
    """
    # Full-width to half-width for alphanumeric (Ａ→A, １→1)
    text = unicodedata.normalize("NFKC", text)
    # Remove punctuation: Japanese (。、！？・「」『』（）) and ASCII
    text = re.sub(r'[。、！？!?.,;:・「」『』（）()【】\[\]{}""''\'\"~〜…ー\-－—\s]', '', text)
    # Lowercase for any romaji/English
    text = text.lower()
    # Convert digits to Japanese words (5000 → ごせん) before hiragana conversion
    text = _digits_to_japanese(text)
    # Convert everything to hiragana (kanji → hiragana, katakana → hiragana)
    text = _to_hiragana(text)
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
        return {
            "cer": {"edits": len(prediction), "ref_len": 0},
            "wer": {"edits": len(prediction), "ref_len": 0},
        }

    cer_edits, cer_ref_len = _calculate_cer(ground_truth, prediction)
    wer_edits, wer_ref_len = _calculate_wer(ground_truth, prediction)
    return {
        "cer": {"edits": cer_edits, "ref_len": cer_ref_len},
        "wer": {"edits": wer_edits, "ref_len": wer_ref_len},
    }


def _calculate_cer(reference, hypothesis):
    """Character Error Rate: returns (edits, ref_len) for corpus-level aggregation."""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    edits = editdistance.eval(ref_chars, hyp_chars)
    return edits, len(ref_chars)


def _calculate_wer(reference, hypothesis):
    """Word Error Rate: returns (edits, ref_len) for corpus-level aggregation."""
    try:
        tagger = _get_fugashi_tagger()
        ref_words = [word.surface for word in tagger(reference)]
        hyp_words = [word.surface for word in tagger(hypothesis)]
    except (ImportError, RuntimeError):
        # Fallback: treat each character as a word (equivalent to CER)
        ref_words = list(reference)
        hyp_words = list(hypothesis)

    edits = editdistance.eval(ref_words, hyp_words)
    return edits, len(ref_words)


def ccr_jp_cer(results):
    """Aggregate CER: corpus-level (total edits / total ref chars)."""
    if not results:
        return 0.0
    total_edits = sum(r["edits"] for r in results)
    total_ref = sum(r["ref_len"] for r in results)
    return total_edits / max(total_ref, 1)


def ccr_jp_wer(results):
    """Aggregate WER: corpus-level (total edits / total ref words)."""
    if not results:
        return 0.0
    total_edits = sum(r["edits"] for r in results)
    total_ref = sum(r["ref_len"] for r in results)
    return total_edits / max(total_ref, 1)
