#!/usr/bin/env python3
"""Private, research-only GLiNER2.5 Small LoRA train + equivalent baseline comparison.

DO NOT PRESENT THESE PARTIAL-SILVER METRICS AS GOLD F1 OR PRECISION.
The source snippets are copyrighted and remain in a private GitHub repository.
"""
import argparse
import collections
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

MODEL = 'fastino/gliner2.5-small-v1'
BUNDLE_SHA = '542f9750193e77c0853c8cb219b86c3cad130aa62cb50769221ad059910a20d7'
LABELS = [
    'fictional creature species', 'fictional person', 'named fantasy technique',
    'named fictional organization', 'named fictional place',
    'named fictional world concept', 'named magical artifact', 'named rank', 'personal title'
]
STEPS_LABEL = 'fixed_epochs_4_early_stop_silver_dev'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def readrows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def prepare(bundle, out):
    assert bundle.is_file(), 'Private v3 ZIP absent; do not fall back to older/pseudo data'
    data = bundle.read_bytes()
    assert sha(data) == BUNDLE_SHA, 'Private ZIP hash mismatch'
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.testzip() is None
        needed = ['train.jsonl', 'dev.jsonl', 'train_audit.jsonl', 'dev_audit.jsonl',
                  'holdout_teacher_provisional_only.jsonl', 'manifest_v3.json']
        for filename in needed:
            filepath = 'v3/' + filename
            assert filepath in z.namelist(), f'Expected private data: {filename}'
            out.mkdir(parents=True, exist_ok=True)
            (out / filename).write_bytes(z.read(filepath))
        manifest = json.loads(z.read('v3/manifest_v3.json'))
    assert manifest['human_gold_samples'] == 0
    assert manifest['status'] == 'UNREVIEWED_PARTIAL_SILVER_NOT_GOLD'
    assert manifest['training_excludes_dev_book'] and manifest['dev_book'] == 'spellcraft'
    train = readrows(out / 'train.jsonl')
    dev = readrows(out / 'dev.jsonl')
    audits_train = readrows(out / 'train_audit.jsonl')
    audits_dev = readrows(out / 'dev_audit.jsonl')
    holdout = readrows(out / 'holdout_teacher_provisional_only.jsonl')
    assert len(train) == len(audits_train) == manifest['splits']['train']['contexts'] == 2748
    assert len(dev) == len(audits_dev) == manifest['splits']['dev']['contexts'] == 468
    assert len(holdout) == 25
    seen_text = set()
    train_books = set()
    dev_books = set()
    for split, pairs in [('train', zip(train, audits_train)), ('dev', zip(dev, audits_dev))]:
        for row, audit in pairs:
            txt = row['input']
            dig = sha(txt.encode())
            assert dig not in seen_text, 'Duplicate passage train/dev'
            seen_text.add(dig)
            assert audit['split'] == split and audit['human_gold_verified'] is False
            assert len(txt) <= 470 and len(txt) > 130
            assert row['output']['entities']
            entities = collections.defaultdict(set)
            for span in audit['spans']:
                start,end,label = span['start'], span['end'], span['label']
                assert label in LABELS and 0 <= start < end <= len(txt)
                entities[label].add(txt[start:end])
            assert dict((k,sorted(v)) for k,v in entities.items()) == row['output']['entities']
            if split == 'train': train_books.add(audit['book'])
            else: dev_books.add(audit['book'])
    assert train_books.isdisjoint(dev_books)
    assert dev_books == {'spellcraft'}
    assert all(sha(row['input'].encode()) not in seen_text for row in holdout)
    summary = {'source_bundle_sha256': BUNDLE_SHA, 'train_examples': len(train),
               'dev_examples': len(dev), 'holdout_teacher_provisional_positive_examples': len(holdout),
               'train_books': len(train_books), 'dev_books': len(dev_books),
               'human_gold_samples': 0, 'labels': LABELS,
               'risk': 'Partial silver unlabelled true entities may train as implicit negatives'}
    (out / 'preflight.json').write_text(json.dumps(summary, indent=2),encoding='utf-8')
    print('PREFLIGHT_PRIVATE_SILVER_PASS',json.dumps(summary),flush=True)
    return summary


def build_token_aligned_training_files(processor, out):
    """Filter whole unalignable examples with the same processor used by the trainer.
    Never convert a dropped positive annotation into an implicit negative.
    Preserve source/dev/holdout for original-protocol comparisons.
    """
    assert hasattr(processor, 'find_unalignable_entities'), 'Alignment API unavailable'
    report = {'source_data_unchanged': True, 'method': 'gliner2_processor_alignment', 'policy': 'skip_whole_window'}
    for split in ('train', 'dev'):
        rows = readrows(out / (split + '.jsonl'))
        kept = []
        failures = collections.Counter()
        for row in rows:
            bad = processor.find_unalignable_entities(row['input'], row['output']['entities'])
            if bad:
                for item in bad:
                    failures[item['entity_type']] += 1
                continue
            kept.append(row)
        assert len(kept) >= len(rows) * 0.80, f'Too many {split} tokenizer failures'
        assert kept
        target = out / (split + '_token_aligned.jsonl')
        with target.open('w', encoding='utf-8') as handle:
            for row in kept:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        persisted = readrows(target)
        assert len(persisted) == len(kept)
        assert all(not processor.find_unalignable_entities(x['input'], x['output']['entities']) for x in persisted)
        report[split] = {'original_windows': len(rows), 'retained_windows': len(kept),
                         'excluded_whole_windows': len(rows) - len(kept),
                         'unalignable_labels_by_class': dict(sorted(failures.items()))}
    (out / 'alignment_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PRIVATE_V3_TOKEN_ALIGNMENT_PASS', json.dumps(report), flush=True)
    return report

def train(out, num_epochs):
    import torch
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
    assert num_epochs in (3,4,5,6)
    torch.manual_seed(20261110)
    torch.set_num_threads(4)
    model = AutoExtractor.from_pretrained(MODEL)
    config = TrainingConfig(
        output_dir=str(out / 'checkpoints'),
        experiment_name='novellex_silver_v3_research',
        num_epochs=num_epochs, max_steps=-1,
        batch_size=4, eval_batch_size=4, gradient_accumulation_steps=2,
        encoder_lr=1e-5, task_lr=1e-4,
        weight_decay=0.01, warmup_ratio=0.06, scheduler_type='cosine',
        fp16=False, bf16=False,
        num_workers=0, pin_memory=False, fused_optimizer=False,
        eval_strategy='epoch', save_best=True, save_total_limit=2,
        early_stopping=True, early_stopping_patience=2,
        logging_steps=100, report_to_wandb=False, seed=20261110,
        use_lora=True, lora_r=16, lora_alpha=32.0,
        lora_target_modules=['encoder','all_task_heads'], save_adapter_only=True,
    )
    trainer = ExtractorTrainer(model, config)
    build_token_aligned_training_files(trainer.processor, out)
    result = trainer.train(train_data=str(out/'train_token_aligned.jsonl'),
                           eval_data=str(out/'dev_token_aligned.jsonl'))
    best = out/'checkpoints'/'best'
    final = out/'checkpoints'/'final'
    adapter = best if (best/'adapter_model.safetensors').is_file() else final
    assert (adapter/'adapter_model.safetensors').is_file()
    assert (adapter/'adapter_config.json').is_file()
    print('PRIVATE_V3_LORA_TRAIN_COMPLETE', json.dumps({'adapter_dir':str(adapter),'num_epochs_upper_bound':num_epochs,'checkpoint_selection':'silver_eval_loss_NOT_gold'}),flush=True)
    return adapter


def predict(model, text):
    raw = model.extract_entities(text,LABELS,threshold=0.5,include_spans=True)
    result=set();invalid=0
    for label,entries in raw.get('entities',{}).items():
        if label not in LABELS:
            invalid+=len(entries or [])
            continue
        for e in entries or []:
            try:
                start,end=int(e['start']),int(e['end'])
                if start>=end or start<0 or end>len(text) or text[start:end]!=e['text']:
                    invalid+=1;continue
            except (KeyError,ValueError,TypeError,OverflowError):
                invalid+=1;continue
            result.add((label,start,end))
    return result,invalid


def evaluate(model, out, split):
    import torch
    samples=readrows(out/(split+'.jsonl'))
    audits=readrows(out/(split+'_audit.jsonl')) if split!='holdout_teacher_provisional_only' else None
    c=collections.Counter(); start=time.monotonic()
    with torch.inference_mode():
        for i,row in enumerate(samples):
            text=row['input']
            if audits is not None:
                gold={(x['label'],x['start'],x['end']) for x in audits[i]['spans']}
            else:
                gold=set()
                for label,terms in row['output']['entities'].items():
                    for term in terms:
                        at=text.find(term)
                        assert at>=0 and text.find(term,at+1)<0
                        gold.add((label,at,at+len(term)))
            assert gold
            predicted,invalid=predict(model,text)
            c['known']+=len(gold);c['matched']+=len(predicted & gold)
            c['predicted']+=len(predicted);c['unverified']+=len(predicted-gold)
            c['invalid']+=invalid;c['contexts']+=1
    return {'known_positive_recall_partial_silver':round(c['matched']/c['known'],6),
            'matched_known_positives':c['matched'],'total_known_positives':c['known'],
            'model_predictions':c['predicted'],'unverified_not_false_positives':c['unverified'],
            'invalid_offsets_skipped':c['invalid'],'contexts':c['contexts'],
            'seconds_cpu':round(time.monotonic()-start,3)}


def compare(out, adapter, num_epochs):
    import torch
    from gliner2 import AutoExtractor
    from peft import PeftModel
    torch.manual_seed(20261110);torch.set_num_threads(4)
    base=AutoExtractor.from_pretrained(MODEL);base.eval()
    before={name:evaluate(base,out,name) for name in ('dev','holdout_teacher_provisional_only')}
    adapted=PeftModel.from_pretrained(base,str(adapter),is_trainable=False)
    adapted.eval()
    model=adapted.get_base_model()
    after={name:evaluate(model,out,name) for name in ('dev','holdout_teacher_provisional_only')}
    report={'status':'research_silver_NOT_production','base_model':MODEL,
        'source_zip_sha256':BUNDLE_SHA,'num_epochs_upper_bound':num_epochs,
        'adapter_sha256':sha((adapter/'adapter_model.safetensors').read_bytes()),
        'eval_threshold':0.5,'labels':LABELS,'human_gold_count':0,
        'baseline':before,'adapter':after,
        'dev_positive_recall_change':round(after['dev']['known_positive_recall_partial_silver']-before['dev']['known_positive_recall_partial_silver'],6),
        'holdout_positive_recall_change':round(after['holdout_teacher_provisional_only']['known_positive_recall_partial_silver']-before['holdout_teacher_provisional_only']['known_positive_recall_partial_silver'],6),
        'limitations':['NO gold precision or F1. Teacher holdout was reused in prior model selection.',
                       'Book-held-out development labels were created with Spellcraft lexicon visibility.',
                       'Missing real entities in partial silver may train as false negatives.',
                       'This is a research candidate, not a validated improved model.']}
    (out/'comparison_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PRIVATE_V3_BASELINE_METRICS',json.dumps(before),flush=True)
    print('PRIVATE_V3_ADAPTER_METRICS',json.dumps(after),flush=True)
    print('PRIVATE_V3_COMPARISON_COMPLETE',report['holdout_positive_recall_change'],flush=True)
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bundle',type=Path,default=Path('private-data/novellex-silver-v3-20261110.zip'))
    ap.add_argument('--output',type=Path,default=Path('silver_v3_output'))
    ap.add_argument('--epochs',type=int,default=4)
    args=ap.parse_args()
    prepare(args.bundle,args.output)
    adapter=train(args.output,args.epochs)
    compare(args.output,adapter,args.epochs)

if __name__=='__main__':main()
