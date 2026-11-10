"""NovelLex: private real-novel silver training, leakage-safe split, conservative evaluation.

Uses the *actual text* of four books' first five chapters plus their partial
silver spans. Mother of Learning is untouched for training and development.
No precision/F1 is computed on incomplete labels, and no model is declared
better from training loss alone.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

MODEL = "fastino/gliner2.5-small-v1"
MAP = {
    "CHARACTER": "fictional person", "PERSON": "fictional person",
    "CHARACTER_ALIAS": "fictional person",
    "PLACE": "named fictional place", "LOCATION": "named fictional place",
    "ORGANIZATION": "named fictional organization",
    "TITLE": "personal title", "ROLE": "personal title",
    "ITEM": "named magical artifact", "ARTIFACT": "named magical artifact",
    "TECHNOLOGY": "named fictional world concept",
    "CULTIVATION_ENERGY": "named fictional world concept",
    "WORLD_CONCEPT": "named fictional world concept", "CONCEPT": "named fictional world concept",
    "SYSTEM_STAT": "named fictional world concept",
    "CULTIVATION_REALM": "named rank", "RANK": "named rank",
    "SKILL": "named fantasy technique", "TECHNIQUE": "named fantasy technique",
    "CREATURE": "fictional creature species",
}
NAMES = sorted(set(MAP.values()))
EXPECTED_TRAIN_BOOKS = {"nano_machine", "tbate", "primal_hunter", "real_cultivation_world"}
HOLDOUT_BOOK = "mother_of_learning"
WINDOW = 440
STRIDE = 220


def dump_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalized_spans(chapter):
    text = chapter["text"]
    seen = {}
    for s in chapter["spans"]:
        label = MAP.get(s["label"])
        if label is None:
            continue
        a, b = s["start"], s["end"]
        assert isinstance(a, int) and isinstance(b, int)
        assert 0 <= a < b <= len(text) and text[a:b] == s["text"], "Broken source offsets"
        key = (a, b)
        if key in seen and seen[key] != label:
            raise ValueError("Conflicting silver label at identical offsets")
        seen[key] = label
    spans = sorted((a, b, lab) for (a, b), lab in seen.items())
    for i in range(1, len(spans)):
        a, b, lab = spans[i]
        x, y, other = spans[i - 1]
        if a < y and (a, b) != (x, y):
            # Crossing/nested labels create unreliable supervision. Fail rather than corrupt it.
            raise ValueError("Overlapping source silver spans need audit")
    return spans


def snippets(chapter):
    text = chapter["text"]
    gold = normalized_spans(chapter)
    if not gold:
        return []
    starts = sorted({max(0, min(max(0, len(text) - WINDOW),
                    (a // STRIDE) * STRIDE - 90)) for a, _, _ in gold})
    rows = []
    for start in starts:
        end = min(start + WINDOW, len(text))
        # A truncated known mention must never be silently labelled O.
        if any(a < start < b or a < end < b for a, b, _ in gold):
            continue
        local = [(a - start, b - start, label) for a, b, label in gold if start <= a and b <= end]
        if not local:
            continue
        # Only positive-context windows. No fabricated all-negative supervision.
        out = defaultdict(set)
        for a, b, label in local:
            out[label].add(text[start + a:start + b])
        sample = {"input": text[start:end], "output": {"entities": {lab: sorted(terms) for lab, terms in out.items()}}}
        reference = {"spans": [{"start": a, "end": b, "label": label} for a, b, label in local]}
        rows.append((sample, reference))
    return rows


def prepare(bundle, out):
    assert bundle.is_file(), "Private EPUB ZIP is required. No synthetic fallback."
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as z:
        names = set(z.namelist())
        assert {"pilot/chapters.jsonl", "review/teacher_review_sheet.csv"} <= names
        chapters = [json.loads(line) for line in z.read("pilot/chapters.jsonl").decode().splitlines()]
        reviews = list(csv.DictReader(io.StringIO(z.read("review/teacher_review_sheet.csv").decode("utf-8-sig"))))
        epubs = [n for n in names if n.startswith("epubs/") and n.endswith(".epub")]
        assert len(epubs) == 5, "Expected five private EPUBs"
    assert len(chapters) == 25
    assert {r["book"] for r in chapters} == EXPECTED_TRAIN_BOOKS | {HOLDOUT_BOOK}
    assert len({(r["book"], int(r["chapter"])) for r in chapters}) == 25
    assert all(len([r for r in chapters if r["book"] == book]) == 5
               for book in EXPECTED_TRAIN_BOOKS | {HOLDOUT_BOOK})
    sets = {"train": [], "dev": [], "holdout": []}
    refs = {"dev": [], "holdout": []}
    counts = Counter()
    silver = Counter()
    for ch in chapters:
        spans = normalized_spans(ch)
        if ch["book"] == HOLDOUT_BOOK:
            assert not spans, "Holdout labels must not be imported into training"
            continue
        target = "dev" if int(ch["chapter"]) == 5 else "train"
        silver[target] += len(spans)
        items = snippets(ch)
        counts[(target, ch["book"])] += len(items)
        for sample, ref in items:
            sets[target].append(sample)
            if target in refs:
                refs[target].append(ref)
    # Frozen holdout: teacher-provisional positive mentions only; not exhaustive gold.
    for row in reviews:
        if row["book"] != HOLDOUT_BOOK or row.get("status") not in ("teacher_reviewed", "approved"):
            continue
        if row.get("manual_approval", "").strip().lower() in ("reject", "no", "false", "0"):
            continue
        lab = MAP.get(row.get("corrected_category", "").strip() or row["category"])
        # Keep the original 25-positive comparison protocol: omit ambiguous aliases.
        if not lab or row["category"] == "CHARACTER_ALIAS":
            continue
        context, term = row["nearby_context"], row["term"]
        a = context.find(term)
        if a < 0 or context.find(term, a + 1) >= 0:
            continue
        sets["holdout"].append({"input": context, "output": {"entities": {lab: [term]}}})
        refs["holdout"].append({"spans": [{"start": a, "end": a + len(term), "label": lab}]})
    assert len(sets["holdout"]) == 25, f"Holdout changed unexpectedly: {len(sets['holdout'])}"
    assert len(sets["train"]) >= 180 and len(sets["dev"]) >= 35, "Insufficient real-novel windows"
    assert silver["train"] >= 500 and silver["dev"] >= 100, "Not using enough of the annotated real corpus"
    for split in sets:
        assert all(0 < len(s["input"]) <= WINDOW and s["output"] for s in sets[split])
    # Prevent any exact text reuse across splits.
    hashes = {key: {hashlib.sha256(row["input"].encode()).hexdigest() for row in rows}
              for key, rows in sets.items()}
    assert all(not hashes[a] & hashes[b] for a, b in (("train", "dev"), ("train", "holdout"), ("dev", "holdout")))
    for split, rows in sets.items():
        dump_jsonl(out / f"{split}.jsonl", rows)
    for split, rows in refs.items():
        dump_jsonl(out / f"{split}_gold_private.jsonl", rows)
    summary = {
        "source": "actual_first_five_chapters_of_five_user_epubs",
        "real_chapter_count": 25, "training_chapters": 16, "dev_chapters": 4,
        "holdout_chapters_used_for_training": 0,
        "train_windows": len(sets["train"]), "dev_windows": len(sets["dev"]),
        "holdout_known_positive_contexts": len(sets["holdout"]),
        "source_silver_spans_train": silver["train"], "source_silver_spans_dev": silver["dev"],
        "train_book_windows": {book: counts[("train", book)] for book in sorted(EXPECTED_TRAIN_BOOKS)},
        "dataset_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "warning": "Silver labels and teacher holdout are incomplete; precision/F1 are not valid. Full novel books are NOT available here.",
    }
    (out / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("REAL_DATA_VALIDATED", json.dumps({k: v for k, v in summary.items() if k != "dataset_sha256"}), flush=True)
    return summary


def finetune(out, steps):
    import torch
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
    torch.set_num_threads(4)
    torch.manual_seed(20261110)
    model = AutoExtractor.from_pretrained(MODEL)
    config = TrainingConfig(
        output_dir=str(out / "checkpoints"), num_epochs=2, max_steps=steps,
        batch_size=2, gradient_accumulation_steps=1,
        encoder_lr=5e-6, task_lr=5e-5, weight_decay=0.01,
        fp16=False, bf16=False, num_workers=0, pin_memory=False,
        fused_optimizer=False, eval_strategy="no", save_best=False,
        logging_steps=40, report_to_wandb=False, seed=20261110,
        use_lora=True, lora_r=16, lora_alpha=32.0,
        lora_target_modules=["encoder", "all_task_heads"], save_adapter_only=True,
    )
    trainer = ExtractorTrainer(model, config)
    trainer.train(train_data=str(out / "train.jsonl"))
    final = out / "checkpoints" / "final"
    assert (final / "adapter_model.safetensors").is_file()
    assert (final / "adapter_config.json").is_file()
    print("REAL_NOVEL_LORA_SAVED", steps, flush=True)
    return final


def predict(model, text):
    raw = model.extract_entities(text, NAMES, threshold=0.5, include_spans=True)
    found = set()
    for lab, ents in raw.get("entities", {}).items():
        assert lab in NAMES
        for e in ents or []:
            a, b = e["start"], e["end"]
            assert isinstance(a, int) and isinstance(b, int) and 0 <= a < b <= len(text)
            assert text[a:b] == e["text"]
            found.add((lab, a, b))
    return found


def evaluate(model, out, split):
    import torch
    rows = load_jsonl(out / f"{split}.jsonl")
    refs = load_jsonl(out / f"{split}_gold_private.jsonl")
    assert len(rows) == len(refs)
    count = Counter()
    start = time.monotonic()
    with torch.inference_mode():
        for row, ref in zip(rows, refs):
            gold = {(e["label"], e["start"], e["end"]) for e in ref["spans"]}
            assert gold
            found = predict(model, row["input"])
            count["known_positive"] += len(gold)
            count["recovered"] += len(gold & found)
            count["predictions"] += len(found)
            count["unverified"] += len(found - gold)
            count["contexts"] += 1
    return {
        "known_positive_recall_provisional": round(count["recovered"] / count["known_positive"], 6),
        "matched_known_positives": count["recovered"],
        "total_known_positives": count["known_positive"],
        "predicted_mentions": count["predictions"],
        "unverified_predictions_not_false_positives": count["unverified"],
        "contexts": count["contexts"],
        "cpu_seconds": round(time.monotonic() - start, 3),
    }


def compare(out, adapter):
    import torch
    from gliner2 import AutoExtractor
    from peft import PeftModel
    cfg = adapter / "adapter_config.json"
    weights = adapter / "adapter_model.safetensors"
    assert cfg.is_file() and weights.is_file()
    assert json.loads(cfg.read_text())["base_model_name_or_path"] == MODEL
    torch.set_num_threads(4)
    torch.manual_seed(20261110)
    base = AutoExtractor.from_pretrained(MODEL)
    base.eval()
    before = {split: evaluate(base, out, split) for split in ("dev", "holdout")}
    adapted = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    adapted.eval()
    after = {split: evaluate(adapted.get_base_model(), out, split) for split in ("dev", "holdout")}
    result = {
        "status": "real_novel_private_baseline_vs_lora_comparison",
        "base_model": MODEL, "threshold": 0.5,
        "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "baseline": before, "fine_tuned": after,
        "delta_holdout_known_positive_recall": round(after["holdout"]["known_positive_recall_provisional"] - before["holdout"]["known_positive_recall_provisional"], 6),
        "delta_dev_known_positive_recall": round(after["dev"]["known_positive_recall_provisional"] - before["dev"]["known_positive_recall_provisional"], 6),
        "limitations": [
            "Training corpus covers only five chapters per book, not full novels.",
            "Source labels are partially silver and holdout is AI-teacher-provisional positive mentions.",
            "Unverified predictions are NOT reliably false positives; no precision/F1 or gold generalization claim.",
            "Validation consists of chapter 5 of the four training books; Mother of Learning never trained on.",
        ],
    }
    (out / "comparison_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("BASELINE_METRICS", json.dumps(before), flush=True)
    print("FINETUNED_METRICS", json.dumps(after), flush=True)
    print("REAL_NOVEL_COMPARISON_COMPLETE", result["delta_holdout_known_positive_recall"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prepare", "train", "compare", "all"))
    ap.add_argument("--bundle", type=Path, default=Path("private-data/novellex-private-input.zip"))
    ap.add_argument("--output", type=Path, default=Path("real_v2_output"))
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=256)
    a = ap.parse_args()
    assert 16 <= a.steps <= 512
    if a.mode in ("prepare", "all"):
        prepare(a.bundle, a.output)
    if a.mode in ("train", "all"):
        assert (a.output / "train.jsonl").is_file(), "Run prepare first"
        adapter = finetune(a.output, a.steps)
    else:
        adapter = a.adapter
    if a.mode in ("compare", "all"):
        assert adapter is not None
        compare(a.output, adapter)


if __name__ == "__main__":
    main()
