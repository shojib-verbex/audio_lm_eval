from lmms_eval.tasks.ccr_jp.utils import (
    _calculate_cer,
    _calculate_wer,
    _normalize_japanese,
)
from loguru import logger as eval_logger


def reazonspeech_doc_to_audio(doc):
    """Load audio from HuggingFace ReazonSpeech dataset (auto-decoded)."""
    return [doc["audio"]]


def reazonspeech_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    default_prompt = "この日本語の音声を正確に書き起こしてください。日本語のみで出力してください。"
    return f"{pre_prompt}{default_prompt}{post_prompt}"


def reazonspeech_process_results(doc, results):
    """Calculate CER and WER for Japanese transcription."""
    ground_truth = doc.get("transcription", "").strip()
    if not ground_truth:
        return {
            "cer": {"edits": 0, "ref_len": 0},
            "wer": {"edits": 0, "ref_len": 0},
        }

    prediction = results[0].strip() if isinstance(results[0], str) else str(results[0])

    ground_truth_norm = _normalize_japanese(ground_truth)
    prediction_norm = _normalize_japanese(prediction)

    if not ground_truth_norm:
        return {
            "cer": {"edits": len(prediction_norm), "ref_len": 0},
            "wer": {"edits": len(prediction_norm), "ref_len": 0},
        }

    cer_edits, cer_ref_len = _calculate_cer(ground_truth_norm, prediction_norm)
    wer_edits, wer_ref_len = _calculate_wer(ground_truth_norm, prediction_norm)
    return {
        "cer": {"edits": cer_edits, "ref_len": cer_ref_len},
        "wer": {"edits": wer_edits, "ref_len": wer_ref_len},
    }


def reazonspeech_cer(results):
    """Aggregate CER: corpus-level (total edits / total ref chars)."""
    if not results:
        return 0.0
    total_edits = sum(r["edits"] for r in results)
    total_ref = sum(r["ref_len"] for r in results)
    return total_edits / max(total_ref, 1)


def reazonspeech_wer(results):
    """Aggregate WER: corpus-level (total edits / total ref words)."""
    if not results:
        return 0.0
    total_edits = sum(r["edits"] for r in results)
    total_ref = sum(r["ref_len"] for r in results)
    return total_edits / max(total_ref, 1)
