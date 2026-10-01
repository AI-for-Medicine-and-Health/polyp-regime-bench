import json,itertools,re,statistics as st
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
a=json.load(open('analysis.json'))
DS=['kvasir_seg','polypgen_wli','polypdb_wli']; DSN=['Kvasir-SEG','PolypGen WLI','PolypDB WLI']
C=['scratch_base','pretrained_base','pretrained_repeat5x','pretrained_aug3x','pretrained_aug5x','pretrained_aug10x']
CN=['Scratch','Pretrained','Repeat-5×','Aug-3×','Aug-5×','Aug-10×']
MN={'fasterrcnn_resnet50_fpn':'Faster R-CNN','yolov3_tinyu':'YOLOv3-TinyU','yolov3_sppu':'YOLOv3-SPPU','yolov5_s':'YOLOv5-S','yolov8_s':'YOLOv8-S','yolov9_s':'YOLOv9-S','yolov10_s':'YOLOv10-S','yolo11_s':'YOLO11-S','yolo12_s':'YOLO12-S','yolo26_s':'YOLO26-S','rtdetr_l':'RT-DETR-L','rtdetr_x':'RT-DETR-X'}
order=list(MN)
# ---- Fig 2: rank heatmap
fig,axes=plt.subplots(1,3,figsize=(7.2,3.6),sharey=True)
for ax,ds,dn in zip(axes,DS,DSN):
    R=np.zeros((12,6))
    for j,c in enumerate(C):
        mm=a['models'][f'{ds}|{c}']; srt=sorted(mm,key=mm.get,reverse=True)
        for i,m in enumerate(order): R[i,j]=srt.index(m)+1
    im=ax.imshow(R,cmap='RdYlGn_r',vmin=1,vmax=12,aspect='auto')
    for i in range(12):
        for j in range(6):
            ax.text(j,i,int(R[i,j]),ha='center',va='center',fontsize=6.5,color='black',fontweight='bold' if order[i]=='rtdetr_x' else 'normal')
    ax.set_xticks(range(6)); ax.set_xticklabels(CN,rotation=45,ha='right')
    ax.axvline(0.5,color='k',lw=1.2)
    t1=[a['tau'][f'{ds}|scratch_base|{c}'][0] for c in C[1:]]
    t2=[a['tau'][f'{ds}|{x}|{y}'][0] for x,y in itertools.combinations(C[1:],2)]
    ax.set_title(f'{dn}\nτ(scratch, pretrained): {min(t1):.2f} to {max(t1):.2f}\nτ(within pretrained): {min(t2):.2f} to {max(t2):.2f}',fontsize=7)
axes[0].set_yticks(range(12)); axes[0].set_yticklabels([MN[m] for m in order])
for lab in axes[0].get_yticklabels():
    if lab.get_text()=='RT-DETR-X': lab.set_fontweight('bold')
cb=fig.colorbar(im,ax=axes,fraction=0.02,pad=0.02); cb.set_label('Rank by test mAP$_{50:95}$ (1 = best)'); cb.ax.invert_yaxis()
fig.savefig('figs/fig2_rank_by_regime.pdf',bbox_inches='tight')
# ---- Fig 3: ensemble gains + cost
lat={}
for line in open('generated/table5_inference_cost.tex'):
    p=[x.strip() for x in line.replace('\\\\','').split('&')]
    if len(p)==6:
        try: lat[p[0]]=float(p[5])
        except: pass
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.2,2.9),gridspec_kw={'width_ratios':[1.35,1]})
P=C[1:]; x=np.arange(len(P)); w=0.26; cols=['#1f77b4','#d95f02','#1b9e77']
for k,(ds,dn) in enumerate(zip(DS,DSN)):
    dw=[];dv=[]
    for c in P:
        mm=a['models'][f'{ds}|{c}']; b=max(mm.values())
        dw.append(a['wbf'][f'{ds}|{c}'][0]-b)
        v=a['vote'].get(f'{ds}|{c}'); dv.append(v['mAP50_95']-b if v else np.nan)
    ax1.bar(x+(k-1)*w,dw,w,color=cols[k],label=dn)
    ax1.scatter(x+(k-1)*w,dv,marker='x',color='k',s=14,zorder=3,label='Hard vote' if k==0 else None)
ax1.axhline(0,color='k',lw=0.8); ax1.set_xticks(x); ax1.set_xticklabels(CN[1:])
ax1.set_ylabel('ΔmAP$_{50:95}$'); ax1.set_title('(a) Ensemble gain over the best individual detector',fontsize=8)
ax1.legend(fontsize=6.5,frameon=False,loc='upper left',ncol=2)
# accuracy-latency
for m in order:
    ys=[a['models'][f'{ds}|pretrained_aug10x'][m] for ds in DS]
    key={'fasterrcnn_resnet50_fpn':'Faster R-CNN R50-FPN'}.get(m,MN[m])
    if key in lat:
        ax2.scatter(lat[key],st.mean(ys),s=14,color='#555555',zorder=3)
        if m in ('rtdetr_x','rtdetr_l','yolo26_s','yolov3_tinyu','fasterrcnn_resnet50_fpn','yolov9_s'):
            off={'rtdetr_l':(3,-7),'yolov9_s':(3,2),'yolo26_s':(-30,4)}.get(m,(3,2))
            ax2.annotate(MN[m],(lat[key],st.mean(ys)),fontsize=5.5,xytext=off,textcoords='offset points')
ens_lat=st.mean(v for k,v in lat.items() if 'WBF' in k)
vote_lat=st.mean(v for k,v in lat.items() if 'vote' in k)
wy=st.mean(a['wbf'][f'{ds}|pretrained_aug10x'][0] for ds in DS)
vy=st.mean(a['vote'][f'{ds}|pretrained_aug10x']['mAP50_95'] for ds in DS)
ax2.scatter([ens_lat],[wy],marker='*',s=70,color='#d62728',zorder=4); ax2.annotate('Top-6 WBF',(ens_lat,wy),fontsize=6,xytext=(-44,3),textcoords='offset points')
ax2.scatter([vote_lat],[vy],marker='x',s=30,color='k',zorder=4); ax2.annotate('Top-6 vote',(vote_lat,vy),fontsize=6,xytext=(-44,-9),textcoords='offset points')
ax2.axvline(1000/30,color='gray',ls='--',lw=0.8); ax2.text(1000/30*1.05,0.618,'33 ms\n(30 FPS)',fontsize=6,color='gray')
ax2.set_xscale('log'); ax2.set_xlim(8,220); ax2.set_xlabel('Latency per frame (ms, log scale)'); ax2.set_ylabel('Mean mAP$_{50:95}$ (Aug-10×)')
ax2.set_title('(b) Accuracy versus inference cost',fontsize=8)
fig.tight_layout(); fig.savefig('figs/fig3_ensemble_gain_cost.pdf',bbox_inches='tight')
print(lat, ens_lat, vote_lat, wy, vy)
