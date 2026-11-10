"""Compare baseline and private LoRA on held-out provisional positive mentions.
These labels are PARTIAL. Recall is only provisional; precision/F1 are not valid.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import torch
from gliner2 import AutoExtractor
from peft import PeftModel

from private_train import MODEL, LABELS, prepare

NAMES = sorted(set(LABELS.values()))


def predict(model, text, threshold):
    raw = model.extract_entities(text, NAMES, threshold=threshold, include_spans=True)
    entities = raw.get("entities", {})
    spans = set()
    for label, entries in entities.items():
        for entry in entries or []:
            a, b = entry["start"], entry["end"]
            if 0 <= a < b <= len(text) and text[a:b] == entry["text"]:
                spans.add((label, a, b))
    return spans


def evaluate(model, rows, threshold):
    count = Counter()
    start = time.monotonic()
    with torch.inference_mode():
        for row in rows:
            text = row["input"]
            positives = set()
            for label, terms in row["output"]["entities"].items():
                for term in terms:
                    pos = text.find(term)
                    assert pos >= 0 and text.find(term, pos + 1) < 0
                    positives.add((label, pos, pos + len(term)))
            guessed = predict(model, text, threshold)
            count["known_positive"] += len(positives)
            count["known_positive_detected"] += len(guessed & positives)
            count["total_predictions"] += len(guessed)
            count["unverified_predictions"] += len(guessed - positives)
            count["contexts"] += 1
    return {
        "known_positive_recall_provisional": round(count["known_positive_detected"] / count["known_positive"], 6),
        "known_positive_detected": count["known_positive_detected"],
        "known_positive_total": count["known_positive"],
        "predicted_mentions": count["total_predictions"],
        "unverified_predictions_not_necessarily_false": count["unverified_predictions"],
        "contexts": count["contexts"],
        "seconds_cpu": round(time.monotonic() - start, 3),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--bundle", type=Path, default=Path("private-data/novellex-private-input.zip"))
    p.add_argument("--output", type=Path, default=Path("comparison_output"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare(args.bundle, args.output)
    rows = [json.loads(x) for x in (args.output / "holdout_provisional.jsonl").read_text().splitlines()]
    assert len(rows) >= 3
    cfg = args.adapter / "adapter_config.json"
    weights = args.adapter / "adapter_model.safetensors"
    assert cfg.is_file() and weights.is_file()
    assert json.loads(cfg.read_text())["base_model_name_or_path"] == MODEL
    torch.set_num_threads(4)
    torch.manual_seed(20261110)
    base = AutoExtractor.from_pretrained(MODEL)
    base.eval()
    baseline = evaluate(base, rows, 0.5)
    adapted = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    adapted.eval()
    trained = evaluate(adapted.get_base_model(), rows, 0.5)
    result = {
        "status": "private_provisional_positive_holdout_comparison",
        "model": MODEL, "threshold": 0.5,
        "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "baseline": baseline, "trained": trained,
        "delta_known_positive_recall": round(
            trained["known_positive_recall_provisional"] -
            baseline["known_positive_recall_provisional"], 6),
        "limitations": [
            "Teacher-provisional partial labels are not an independently audited gold set.",
            "Unlabeled predictions are not necessarily false positives.",
            "Precision, F1 and false-positive rates cannot be established.",
            "Holdout contains selected positive snippets, not full chapters or representative negatives.",
            "CPU inference timing on GitHub runners is not Android performance.",
        ],
    }
    (args.output / "comparison_report.json").write_text(json.dumps(result, indent=2))
    print("BASELINE", json.dumps(baseline), flush=True)
    print("PRIVATE_LORA", json.dumps(trained), flush=True)
    print("COMPARISON_COMPLETE", result["delta_known_positive_recall"], flush=True)


if __name__ == "__main__":
    main()
