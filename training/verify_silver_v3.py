#!/usr/bin/env python3
"""Self-contained structural audit; can run inside private GitHub Actions runner.
Never verifies semantic truth. Never reads or prints private snippet text.
"""
import argparse
import collections
import hashlib
import io
import json
import zipfile

PARSER=argparse.ArgumentParser()
PARSER.add_argument('--zip',required=True)
PARSER.add_argument('--expected-sha256',default=None)

def go():
 args=PARSER.parse_args()
 data=open(args.zip,'rb').read();h=hashlib.sha256(data).hexdigest()
 if args.expected_sha256:assert h==args.expected_sha256, 'dataset source SHA mismatch'
 with zipfile.ZipFile(io.BytesIO(data)) as z:
  assert z.testzip() is None
  def read(p):return [json.loads(s) for s in z.read(p).decode('utf-8').splitlines() if s.strip()]
  m=json.loads(z.read('v3/manifest_v3.json'))
  assert m['human_gold_samples']==0 and 'NOT_GOLD' in m['status']
  assert m['source_package_sha256']=='d82c5434e01a9cd417a3f6637b56ebb403ea1f41e9500f132bdecfa313096729'
  assert m['epub_verified_sha256_count']==10 and m['chapter_sha256_verified_count']==717
  splits=('train','dev','tr_train','tr_dev')
  sourcechapters={};all_ids=set();all_text=set();totals={}
  for split in splits:
   examples=read('v3/'+split+'.jsonl')
   audits=read('v3/'+split+'_audit.jsonl')
   assert len(examples)==len(audits)==m['splits'][split]['contexts']
   nspans=0
   for e,a in zip(examples,audits):
    assert a['id'] not in all_ids;all_ids.add(a['id'])
    assert a['split']==split and a['human_gold_verified'] is False
    assert a['curation']=='audit_guided_silver_v3_not_gold'
    text=e['input'];digest=hashlib.sha256(text.encode()).hexdigest()
    assert digest not in all_text;all_text.add(digest)
    assert 130<len(text)<=470 and a['chapter_end']-a['chapter_start']==len(text)
    assert e['output']['entities'] and a['spans']
    starts=set(); group=collections.defaultdict(set)
    for s in a['spans']:
     start,end,lab=s['start'],s['end'],s['label']
     assert isinstance(start,int) and isinstance(end,int) and 0<=start<end<=len(text)
     assert (start,end) not in starts;starts.add((start,end))
     group[lab].add(text[start:end]);nspans+=1
    assert all(span1['end']<=span2['start'] or span2['end']<=span1['start'] for i,span1 in enumerate(a['spans']) for span2 in a['spans'][i+1:])
    assert {l:sorted(terms) for l,terms in group.items()}==e['output']['entities']
    for label,terms in group.items():
     for term in terms:
      assert text.count(term)==sum(text[s['start']:s['end']]==term for s in a['spans'])
    sourcechapters.setdefault(split,set()).add((a['book'],a['chapter']))
    if split in ('train','dev'):
     assert a['language']=='en'
    else:
     assert a['language']=='tr'
    assert a['book']!='mother_of_learning'
   assert nspans==m['splits'][split]['mentions']
   totals[split]={'rows':len(examples),'spans':nspans,'books':len(m['splits'][split]['books'])}
  trainbooks={x[0] for x in sourcechapters['train']}
  devbooks={x[0] for x in sourcechapters['dev']}
  assert trainbooks.isdisjoint(devbooks) and devbooks=={'spellcraft'}
  hold=read('v3/holdout_teacher_provisional_only.jsonl')
  assert len(hold)==25
  assert all(hashlib.sha256(x['input'].encode()).hexdigest() not in all_text for x in hold)
  print(json.dumps({'result':'PASS_structural_NOT_semantic','sha256':h,'splits':totals,'holdout_provisional':len(hold),'human_gold_count':0},sort_keys=True))
if __name__=='__main__':go()
