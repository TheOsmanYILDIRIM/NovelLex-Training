# NovelLex Training (PRIVATE)

Private training of **GLiNER2.5 Small** for fantasy terms in English novel passages.

## Data actually available
The private ZIP at \`private-data/novellex-private-input.zip\` contains the **first five chapters** of five user-supplied EPUBs (not complete novels), 884 partial silver spans across four books, and a teacher-provisional Mother of Learning holdout. Never copy these texts or derived examples to a public repository.

## v2 workflow
The **real-novel v2** workflow on \`main\` trains LoRA on 16 chapters / 337 positive context windows, validates on chapter 5 of each of those four source books (4 chapters / 97 context windows), and compares the frozen baseline against the adapter on 25 positive contexts from the completely withheld fifth book, Mother of Learning. The exact row counts are enforced and reported in the workflow. All 884 available pilot silver spans are validated: 668 from training chapters, 216 from validation chapters; never use validation/holdout labels for weight updates.

The new LoRA uses 256 training steps, rank 16, reduced learning rates, and positive-context windows rather than training on 75 snippets. Evaluation prints aggregate counts only, never private text. The model artifact stays in this PRIVATE repository and has a short retention period.

**No legitimate precision, false-positive rate, or F1 is available on this partial silver / provisional teacher data.** We report recall of *known positives*, total predictions, unverified predictions, and dev/holdout deltas. Do not deploy unless the independent holdout comparison shows real value; for trustworthy F1, manually adjudicate complete labels and representative negatives.

## Older runs
The synthetic LoRA run did not improve F1. A private 64-step LoRA trained on 75 provisional contexts scored 19/25 against the base model's 21/25 known positives on the withheld book, and is **not** a successful improvement. v2 is a distinct attempt to remedy tiny training data and excess effective epochs.

## Privacy
Private is access-controlled, **not encrypted**. People with repository access can view private files and Actions artifacts. Keep GitHub Actions logs free of original text, tokens, and annotations. No public releases or model uploads.

## Verified v2 outcomes — 2026-11-10

- [Real-novel fine-tune + nine-label benchmark](https://github.com/TheOsmanYILDIRIM/NovelLex-Training/actions/runs/42872631001) succeeded: 256 LoRA steps, 337 real-text training windows, 97 development windows, withheld Mother of Learning 25 positive contexts. Development known-positive recall: base **207/319 (64.89%)**, adapted **225/319 (70.53%)**. Withheld book under this nine-label schema: base **19/25 (76%)**, adapted **21/25 (84%)**.
- [Exact original eight-label benchmark](https://github.com/TheOsmanYILDIRIM/NovelLex-Training/actions/runs/42874164506) succeeded **without retraining**: base **21/25 (84%)**, adapted v2 **21/25 (84%)**. Earlier small private adapter: **19/25 (76%)**. Thus **v2 recovered the earlier regression but did not beat the original base under the original evaluation protocol.**
- [Private v2 model artifact](https://github.com/TheOsmanYILDIRIM/NovelLex-Training/actions/runs/42872631001/artifacts/13790078974) contains trained adapter, config, data summary and aggregate metrics. Artifacts have limited retention.
- Evaluation uses only partial silver and teacher-provisional labels. Unverified predictions are not verified errors; neither precision nor F1 is established. No Android/model integration or fully audited gold test has been performed. **Retain the base model as the current default** until genuinely independent gold data establishes improved quality.

## 2026-11-10 — audited partial-silver v3 *research* (not gold)

The user-provided 10 EPUBs were re-extracted locally and exactly verified against their SHA256 source inventory: **10/10 original EPUBs, 717/717 chapter hashes and 3729/3729 original text slices** matched. A conservative v3 pass abstained on 184 ambiguous terms, retagged 26 audited candidates, discarded 41 teacher-only uncertain mentions and created an offset-addressable 300-mention pending manual review queue. **Human-verified gold count is zero.**

Private training package: `private-data/novellex-silver-v3-20261110.zip` (SHA256 `542f9750193e77c0853c8cb219b86c3cad130aa62cb50769221ad059910a20d7`). English source-faithful **2748 train** / **468 book-disjoint Spellcraft dev** candidates; 380 Turkish train / 100 Turkish dev kept separate; 25 *Mother of Learning* old teacher-provisional positives never trained on. **This is STILL incompletely labeled partial silver.** Unknown true entities may become implicit negatives in standard GLiNER training.

Workflow `private-silver-v3.yml` runs 4-epoch LoRA research on GitHub-hosted CPU (not GPU) and compares frozen original GLiNER2.5 Small and the adapter at identical label schema and threshold on provisional known-positive recall. Its dev is book-disjoint for weights but its annotations were made with lexicon visibility; the repeatedly seen 25-case holdout is not independent gold. Never report precision/F1, production improvement or Android performance from these checks. No application default model changes.
