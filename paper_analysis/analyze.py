import json,re,statistics as st,itertools
from scipy.stats import kendalltau
G='generated/'
d=json.load(open(G+'results_data_ap001_operating020_vote001.json'))['conditions']
DS=['kvasir_seg','polypgen_wli','polypdb_wli']
CONDS=['scratch_base','pretrained_base','pretrained_repeat5x','pretrained_aug3x','pretrained_aug5x','pretrained_aug10x']
out={'models':{},'valsingle':{},'vote':{},'wbf':{},'tau':{}}
for ds in DS:
  for c in CONDS:
    ms={}
    for s in d[c]:
      for m,v in d[c][s][ds]['models'].items(): ms.setdefault(m,[]).append(v['mAP50_95'])
    out['models'][f'{ds}|{c}']={m:st.mean(x) for m,x in ms.items()}
    vs=[];vo=[]
    for s in d[c]:
      e=d[c][s][ds]; b=e.get('best_single_model_on_validation')
      if b: vs.append(e['models'][b])
      if 'vote_test_metrics' in e and e['vote_test_metrics']: vo.append(e['vote_test_metrics'])
    agg=lambda L,k: st.mean(x[k] for x in L)
    if vs: out['valsingle'][f'{ds}|{c}']={k:agg(vs,k) for k in ['mAP50_95','mAP50','precision','recall','f1']}
    if vo: out['vote'][f'{ds}|{c}']={k:agg(vo,k) for k in ['mAP50_95','mAP50','precision','recall','f1']}
# parse WBF val-tuned from table3
t=open(G+'table3_conditions.tex').read()
cur=None;cond=None
names={'Scratch':'scratch_base','Pretrained base':'pretrained_base','Pretrained Repeat5':'pretrained_repeat5x','Pretrained Aug3':'pretrained_aug3x','Pretrained Aug5':'pretrained_aug5x','Pretrained Aug10':'pretrained_aug10x'}
dsn={'Kvasir-SEG':'kvasir_seg','PolypGen WLI':'polypgen_wli','PolypDB WLI':'polypdb_wli'}
for line in t.splitlines():
  m=re.search(r'textbf\{(Kvasir-SEG|PolypGen WLI|PolypDB WLI)\}',line)
  if m: cur=dsn[m.group(1)];continue
  for k,v in names.items():
    if line.startswith(k+' &'): cond=v
  if 'WBF (val-tuned)' in line and cur:
    nums=re.findall(r'(\d\.\d{3})\\pm(\d\.\d{3})',line)
    out['wbf'][f'{cur}|{cond}']=[float(a) for a,b in nums]
for ds in DS:
  R={}
  for c in CONDS:
    mm=out['models'][f'{ds}|{c}']; R[c]=mm
  ms=sorted(R[CONDS[0]])
  for a,b in itertools.combinations(CONDS,2):
    tau,p=kendalltau([R[a][m] for m in ms],[R[b][m] for m in ms])
    out['tau'][f'{ds}|{a}|{b}']=(tau,p)
json.dump(out,open('analysis.json','w'),indent=1)
for ds in DS:
  print(ds)
  for c in CONDS:
    k=f'{ds}|{c}';mm=out['models'][k];best=max(mm,key=mm.get)
    w=out['wbf'].get(k,[None])[0]; v=out['vote'].get(k,{}).get('mAP50_95'); vs=out['valsingle'].get(k,{}).get('mAP50_95')
    print(f' {c:22s} mean12={st.mean(mm.values()):.3f} best={best}:{mm[best]:.3f} valsingle={vs} vote={v} wbf={w}')
  print(' tau scratch-base %.2f p=%.3f'%out['tau'][f'{ds}|scratch_base|pretrained_base'])
  pt=[out['tau'][f'{ds}|{a}|{b}'][0] for a,b in itertools.combinations(CONDS[1:],2)]
  print(' tau among pretrained min %.2f mean %.2f'%(min(pt),st.mean(pt)))
  sc=[out['tau'][f'{ds}|scratch_base|{b}'][0] for b in CONDS[1:]]
  print(' tau scratch vs pretrained', ['%.2f'%x for x in sc])
