import re,json
G='generated/'; T='tables/'
a=json.load(open('analysis.json'))
DS=['kvasir_seg','polypgen_wli','polypdb_wli']; DSN=['Kvasir-SEG','PolypGen WLI','PolypDB WLI']
REN=[('Val-selected single','Selected single'),('Top-6 hard vote ($\\geq$3)','Hard vote (3 of 6)'),('Top-6 WBF (fixed)','WBF (fixed)'),('Top-6 WBF (val-tuned)','WBF (tuned)'),('WBF (val-tuned)','WBF (tuned)'),
     ('Pretrained Aug10','Aug-10$\\times$'),('Pretrained Aug5','Aug-5$\\times$'),('Pretrained Aug3','Aug-3$\\times$'),('Pretrained Repeat5','Repeat-5$\\times$'),('Pretrained base','Pretrained'),('Faster R-CNN R50-FPN','Faster R-CNN')]
def ren(s):
    for x,y in REN: s=s.replace(x,y)
    return s
def parse(fn):
    out={};cur=None
    for line in open(fn):
        m=re.search(r'textbf\{(Kvasir-SEG|PolypGen WLI|PolypDB WLI)\}',line)
        if m: cur=m.group(1); out[cur]=[]; continue
        nums=re.findall(r'(\d\.\d{3})\\pm(\d\.\d{3})',line)
        if cur and len(nums)==5: out[cur].append((ren(line.split('&')[0].strip()),[float(x) for x,_ in nums]))
    return out
A10=parse(G+'table2_aug10x.tex')
# ---- Table 3 (main Aug10 benchmark, means)
cols=[0,2,3,4]
L=[r'\begin{table*}[t]',r'\centering\footnotesize\setlength{\tabcolsep}{3.2pt}',
r'\caption{Benchmark under the Aug-10$\times$ regime (test means over three seeds). AP is $\mathrm{mAP}_{50:95}$; precision (P), recall (R) and F1 are computed at score $>0.20$ and IoU 0.50. Bold marks the highest value in each column. Standard deviations are given in Table~\ref{tab:A_aug10}.}\label{tab:aug10}',
r'\begin{tabular}{@{}l'+'cccc'*3+'@{}}',r'\toprule',
' & '+' & '.join(r'\multicolumn{4}{c}{%s}'%d for d in DSN)+r' \\',
r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(l){10-13}',
'Detector / ensemble'+' & AP & P & R & F1'*3+r' \\',r'\midrule']
names=[n for n,_ in A10['Kvasir-SEG']]
best={d:[max(v[c] for _,v in A10[d]) for c in cols] for d in DSN}
for i,n in enumerate(names):
    if n=='Selected single': L.append(r'\midrule')
    cells=[]
    for d in DSN:
        v=A10[d][i][1]
        for j,c in enumerate(cols):
            s='%.3f'%v[c]; cells.append(r'\textbf{%s}'%s if abs(v[c]-best[d][j])<1e-9 else s)
    L.append(n+' & '+' & '.join(cells)+r' \\')
L+= [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
open(T+'tab3_aug10_benchmark.tex','w').write('\n'.join(L)+'\n')
# ---- Table 2 (regime summary)
C=['scratch_base','pretrained_base','pretrained_repeat5x','pretrained_aug3x','pretrained_aug5x','pretrained_aug10x']
CN=['Scratch','Pretrained','Repeat-5$\\times$','Aug-3$\\times$','Aug-5$\\times$','Aug-10$\\times$']
L=[r'\begin{table*}[t]',r'\centering\footnotesize\setlength{\tabcolsep}{3.5pt}',
r'\caption{Test $\mathrm{mAP}_{50:95}$ by training regime (means over three seeds). ``Mean'' averages the 12 detectors; ``Best single'' is the highest-scoring individual detector on the test set (an optimistic reference); the RT-DETR-X column gives its score and, in parentheses, its rank among the 12 detectors; WBF is the validation-tuned Top-6 fusion. Full metrics with standard deviations are in Table~\ref{tab:A_conditions}.}\label{tab:regimes}',
r'\begin{tabular}{@{}l'+'cccc'*3+'@{}}',r'\toprule',
' & '+' & '.join(r'\multicolumn{4}{c}{%s}'%d for d in DSN)+r' \\',
r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(l){10-13}',
'Regime'+' & Mean & Best single & RT-DETR-X & WBF'*3+r' \\',r'\midrule']
for c,cn in zip(C,CN):
    cells=[]
    for ds in DS:
        mm=a['models'][f'{ds}|{c}']; mean=sum(mm.values())/12; srt=sorted(mm,key=mm.get,reverse=True)
        b=mm[srt[0]]; rx=mm['rtdetr_x']; rk=srt.index('rtdetr_x')+1; w=a['wbf'][f'{ds}|{c}'][0]
        ws='%.3f'%w; ws=r'\textbf{%s}'%ws if w>b else ws
        cells+=['%.3f'%mean,'%.3f'%b,'%.3f (%d)'%(rx,rk),ws]
    L.append(cn+' & '+' & '.join(cells)+r' \\')
    if c=='scratch_base': L.append(r'\midrule')
L+=[r'\bottomrule',r'\end{tabular}',r'\par\smallskip\parbox{\textwidth}{\footnotesize Bold WBF values exceed the best individual detector.}',r'\end{table*}']
open(T+'tab2_regime_summary.tex','w').write('\n'.join(L)+'\n')
# ---- Table 4 cost
rows=[]
for line in open(G+'table5_inference_cost.tex'):
    p=[x.strip() for x in line.replace('\\\\','').split('&')]
    if len(p)==6 and re.match(r'[\d.]+$',p[1]): rows.append(p)
L=[r'\begin{table}[t]',r'\centering\footnotesize\setlength{\tabcolsep}{3.5pt}',
r'\caption{Inference cost at $640\times640$, batch size 1. Ensemble latency is the serial sum of the six member detectors plus measured fusion time, averaged over the three seed-specific Top-6 selections. FPS $=1000/\text{latency}$. Timings were obtained on a shared NVIDIA H200 NVL and are indicative rather than a controlled hardware benchmark.}\label{tab:cost}',
r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule','Detector / ensemble & Params (M) & GFLOPs & Latency (ms) & FPS \\\\',r'\midrule']
for p in rows:
    n=ren(p[0]).replace('Top-6 Hard vote','hard vote').replace('Top-6 WBF','WBF')
    if 'Kvasir-SEG hard' in n: L.append(r'\midrule')
    L.append(f'{n} & {p[1]} & {p[2]} & {p[5]} & {1000/float(p[5]):.1f} \\\\')
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
open(T+'tab4_inference_cost.tex','w').write('\n'.join(L)+'\n')
# ---- Table 5 errors
s=open(G+'table4_rtdetrx_errors.tex').read()
body=s[s.index(r'\begin{tabular}'):s.index(r'\end{tabular}')+len(r'\end{tabular}')]
L=[r'\begin{table}[t]',r'\centering\footnotesize\setlength{\tabcolsep}{3pt}',
r'\caption{Error decomposition for RT-DETR-X under Aug-10$\times$ (counts per seed, mean $\pm$ SD over three seeds; score $>0.20$, IoU 0.50). FN = pure miss + near miss; FP = near miss + duplicate + other FP. A near miss overlaps a lesion with IoU 0.10--0.49. Test sets contain 112, 105 and 392 lesions, so counts are not comparable as rates across datasets.}\label{tab:errors}',body,r'\end{table}']
open(T+'tab5_rtdetrx_errors.tex','w').write('\n'.join(L)+'\n')
# ---- Table 1 datasets
s=open(G+'table1_datasets.tex').read()
body=s[s.index(r'\begin{tabular}'):s.index(r'\end{tabular}')+len(r'\end{tabular}')]
L=[r'\begin{table}[t]',r'\centering\footnotesize',
r'\caption{Dataset composition (images / annotated boxes). PolypGen uses centres C1--C5 for training and validation and holds out C6 for testing. PolypDB is centre-stratified, so its test set is not a held-out-centre evaluation.}\label{tab:datasets}',body,r'\end{table}']
open(T+'tab1_datasets.tex','w').write('\n'.join(L)+'\n')
# ---- Appendix longtables
FOOT=r'\noindent{\footnotesize Test mean $\pm$ SD over three seeds. AP uses all cached detections with score $\geq0.001$ (WBF: validation-selected input threshold); P, R and F1 use score $>0.20$ and IoU 0.50. Bold marks the best value in each block.}\par\endgroup\medskip'
def appx(src,dst,cap,lab):
    s=open(G+src).read()
    s=re.sub(r'\\caption\{.*?\}\\label\{[^}]*\}',lambda m:r'\caption{%s}\label{%s}'%(cap,lab),s,count=1,flags=re.S)
    s=s[:s.index(r'\noindent{\footnotesize')]+FOOT+'\n'
    s=ren(s).replace('Model / condition','Detector / ensemble')
    open(T+dst,'w').write(s)
appx('table2_aug10x.tex','tabA1_aug10_full.tex','Full Aug-10$\\times$ results with standard deviations.','tab:A_aug10')
appx('table3_conditions.tex','tabA2_regimes_full.tex','Training-regime comparison: 12-detector mean, RT-DETR-X and tuned WBF.','tab:A_conditions')
appx('appendix_table_s1.tex','tabA3_kvasir.tex','Detector-level results on Kvasir-SEG for the Pretrained, Repeat-5$\\times$ and Aug-5$\\times$ regimes.','tab:A_kvasir')
appx('appendix_table_s2.tex','tabA4_polypgen.tex','Detector-level results on PolypGen WLI for the Pretrained, Repeat-5$\\times$ and Aug-5$\\times$ regimes.','tab:A_polypgen')
appx('appendix_table_s3.tex','tabA5_polypdb.tex','Detector-level results on PolypDB WLI for the Pretrained, Repeat-5$\\times$ and Aug-5$\\times$ regimes.','tab:A_polypdb')
