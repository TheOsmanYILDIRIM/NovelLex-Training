"""Private NovelLex real-novel training with explicit partial-label safeguards.

Requires private-data/novellex-private-input.zip uploaded to this PRIVATE repo.
Uses teacher-reviewed positive contexts only. This is provisional silver training,
NOT a fully adjudicated NER dataset; do not treat full unlabeled chapters as O.
"""
import argparse
import csv
import io
import json
import hashlib
from collections import defaultdict
from pathlib import Path
import zipfile

MODEL = "fastino/gliner2.5-small-v1"
LABELS = {
    "CHARACTER": "fictional person", "PERSON": "fictional person",
    "PLACE": "named fictional place", "LOCATION": "named fictional place",
    "ITEM": "named magical artifact", "ARTIFACT": "named magical artifact",
    "SKILL": "named fantasy technique", "TECHNIQUE": "named fantasy technique",
    "ORGANIZATION": "named fictional organization",
    "WORLD_CONCEPT": "named fictional world concept", "CONCEPT": "named fictional world concept",
    "RANK": "named rank", "TITLE": "personal title", "ROLE": "personal title",
}


def prepare(bundle, output):
    if not bundle.exists():
        raise FileNotFoundError("Private input ZIP absent: no training on synthetic fallback")
    with zipfile.ZipFile(bundle) as z:
        names = set(z.namelist())
        required = {"pilot/chapters.jsonl", "pilot/manifest.json", "review/teacher_review_sheet.csv"}
        assert required <= names, f"Missing required input: {required - names}"
        manifest = json.loads(z.read("pilot/manifest.json"))
        chapters = [json.loads(s) for s in z.read("pilot/chapters.jsonl").decode().splitlines()]
        csv_text = z.read("review/teacher_review_sheet.csv").decode("utf-8-sig")
        reviews = list(csv.DictReader(io.StringIO(csv_text)))
        assert len({(r["book"], r["chapter"]) for r in chapters}) == len(chapters)
        books = {r["book"] for r in chapters}
        assert "mother_of_learning" in books and len(books) == 5
        by_chapter = {(r["book"], str(r["chapter"])): r for r in chapters}
        train, holdout = defaultdict(list), defaultdict(list)
        discarded = defaultdict(int)
        for row in reviews:
            if row.get("status") not in {"teacher_reviewed", "approved"}:
                discarded["not_reviewed"] += 1
                continue
            decision = row.get("manual_approval", "").strip().lower()
            if decision in {"no", "reject", "false", "0"}:
                discarded["rejected"] += 1
                continue
            label = row.get("corrected_category", "").strip() or row.get("category", "").strip()
            if label not in LABELS:
                discarded["unknown_category"] += 1
                continue
            # No chapter-wide O labels: training only on selected positive contexts.
            # A row's nearby_context is NOT a complete annotation of all mentions.
            context = row.get("nearby_context", "")
            term = row.get("term", "")
            if not term or not context:
                discarded["no_context"] += 1
                continue
            if (row["book"], str(row.get("chapter", ""))) not in by_chapter:
                # v0.3 sheet uses sample IDs with chapter IDs, not chapter column.
                pass
            start = context.find(term)
            if start < 0 or context.find(term, start + 1) >= 0:
                discarded["ambiguous_context"] += 1
                continue
            entry = {"input": context, "output": {"entities": {LABELS[label]: [term]}}}
            bucket = holdout if row["book"] == "mother_of_learning" else train
            bucket[row["book"]].append(entry)
        # Deduplicate repeated snippets. This is NOT an independent gold set.
        def flatten(group):
            unique = {}
            for book, rows in group.items():
                for row in rows:
                    key = (book, row["input"], json.dumps(row["output"], sort_keys=True))
                    unique[key] = row
            return list(unique.values())
        training, heldout = flatten(train), flatten(holdout)
        assert len(training) >= 25, "Insufficient reviewed private positive snippets"
        assert len(heldout) >= 3, "Missing Mother of Learning private holdout"
        assert not {r["input"] for r in training} & {r["input"] for r in heldout}
        output.mkdir(parents=True, exist_ok=True)
        for name, rows in (("train", training), ("holdout_provisional", heldout)):
            with (output / f"{name}.jsonl").open("w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {"source": "private_epub_teacher_provisional", "train_snippets": len(training),
                   "holdout_snippets": len(heldout), "books": sorted(books),
                   "discarded": dict(discarded), "source_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                   "warning": "Partial AI-teacher positives; NOT complete gold. Holdout is never trained on."}
        (output / "checks.json").write_text(json.dumps(summary, indent=2))
        print("PRIVATE_DATA_VALIDATED", json.dumps(summary), flush=True)
        return summary


def train(output, steps):
    import torch
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
    torch.set_num_threads(4)
    model = AutoExtractor.from_pretrained(MODEL)
    cfg = TrainingConfig(output_dir=str(output / "checkpoints"), num_epochs=1,
        max_steps=steps, batch_size=2, encoder_lr=1e-5, task_lr=1e-4,
        fp16=False, bf16=False, num_workers=0, pin_memory=False,
        fused_optimizer=False, eval_strategy="no", save_best=False,
        logging_steps=10, report_to_wandb=False, seed=20261110,
        use_lora=True, lora_r=8, lora_alpha=16.0,
        lora_target_modules=["encoder", "all_task_heads"], save_adapter_only=True)
    trainer = ExtractorTrainer(model, cfg)
    trainer.train(train_data=str(output / "train.jsonl"))
    final = output / "checkpoints" / "final"
    assert list(final.glob("*.safetensors")), "No trained adapter saved"
    (output / "training_report.json").write_text(json.dumps({
        "status": "provisional_private_training", "steps_requested": steps,
        "model": MODEL, "warning": "Private provisional silver data; real accuracy not established."}, indent=2))
    print("PRIVATE_ADAPTER_SAVED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "train"))
    parser.add_argument("--bundle", type=Path, default=Path("private-data/novellex-private-input.zip"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()
    assert 1 <= args.steps <= 512
    prepare(args.bundle, args.output)
    if args.mode == "train":
        train(args.output, args.steps)
