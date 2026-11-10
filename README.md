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
