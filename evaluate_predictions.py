"""
Evaluate existing prediction manifest files using our normalization pipeline.

Usage:
    python evaluate_predictions.py --pred_manifest /path/to/pred.jsonl
    python evaluate_predictions.py --pred_dir /path/to/dir  # evaluate all *_pred.jsonl files
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from lmms_eval.tasks.ccr_jp.utils import _normalize_japanese, _calculate_cer, _calculate_wer


def evaluate_manifest(manifest_path):
    total_cer_edits = 0
    total_cer_ref = 0
    total_wer_edits = 0
    total_wer_ref = 0
    total_duration = 0.0
    count = 0
    skipped = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            ref = sample.get("text", "").strip()
            hyp = sample.get("pred_text", "").strip()
            duration = float(sample.get("duration", 0))

            if not ref or not hyp:
                skipped += 1
                continue

            ref_norm = _normalize_japanese(ref)
            hyp_norm = _normalize_japanese(hyp)

            if not ref_norm:
                skipped += 1
                continue

            cer_edits, cer_ref_len = _calculate_cer(ref_norm, hyp_norm)
            wer_edits, wer_ref_len = _calculate_wer(ref_norm, hyp_norm)

            total_cer_edits += cer_edits
            total_cer_ref += cer_ref_len
            total_wer_edits += wer_edits
            total_wer_ref += wer_ref_len
            total_duration += duration
            count += 1

    cer = (total_cer_edits / max(total_cer_ref, 1)) * 100
    wer = (total_wer_edits / max(total_wer_ref, 1)) * 100

    return {
        "file": os.path.basename(manifest_path),
        "samples": count,
        "skipped": skipped,
        "hours": total_duration / 3600,
        "cer": cer,
        "wer": wer,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction manifests with our normalization pipeline")
    parser.add_argument("--pred_manifest", type=str, help="Path to a single prediction manifest")
    parser.add_argument("--pred_dir", type=str, help="Directory containing *_pred.jsonl files")
    args = parser.parse_args()

    manifests = []
    if args.pred_manifest:
        manifests.append(args.pred_manifest)
    elif args.pred_dir:
        for f in sorted(os.listdir(args.pred_dir)):
            if f.endswith("_pred.jsonl"):
                manifests.append(os.path.join(args.pred_dir, f))
    else:
        parser.error("Provide --pred_manifest or --pred_dir")

    print(f"{'Dataset':<55} | {'Hrs':>5} | {'Samples':>7} | {'CER':>6} | {'WER':>6}")
    print("-" * 55 + "-|-------|---------|-" + "-" * 6 + "-|-" + "-" * 6)

    all_results = []
    for path in manifests:
        print(f"Processing {os.path.basename(path)}...", end=" ", flush=True)
        result = evaluate_manifest(path)
        all_results.append(result)
        print("done")
        print(f"{result['file']:<55} | {result['hours']:>5.2f} | {result['samples']:>7,} | {result['cer']:>5.2f}% | {result['wer']:>5.2f}%")

    if len(all_results) > 1:
        total_samples = sum(r["samples"] for r in all_results)
        total_hours = sum(r["hours"] for r in all_results)
        avg_cer = sum(r["cer"] for r in all_results) / len(all_results)
        avg_wer = sum(r["wer"] for r in all_results) / len(all_results)
        print("-" * 55 + "-|-------|---------|-" + "-" * 6 + "-|-" + "-" * 6)
        print(f"{'--- Average ---':<55} | {total_hours:>5.2f} | {total_samples:>7,} | {avg_cer:>5.2f}% | {avg_wer:>5.2f}%")


if __name__ == "__main__":
    main()
