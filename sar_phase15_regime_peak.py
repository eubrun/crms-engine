"""Phase15: causal regime-peak exit after Daily-SAR BUY.
Question: is current running peak likely to be exceeded materially before the event horizon ends?
Uses price acceleration, drawdown/peak age and lower-TF SAR deterioration. Chronological OOF.
"""
exec(open('sar_phase13b_exit_fast.py').read().split("# 90+ meaningful configurations")[0])
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROWS=[]; META=[]
for ei,d in enumerate(EV):
 e=d['e'];hi=d['hi'];lo=d['lo'];cl=d['cl'];ts=d['ts'];n=min(len(ts),240);peak=e;peakj=0
 for j in range(6,n-1):
  if hi[j]>=peak:peak=hi[j];peakj=j
  runup=peak/e-1;dd=cl[j]/peak-1;age=j-peakj
  r1=cl[j]/cl[j-1]-1;r3=cl[j]/cl[j-3]-1;r6=cl[j]/cl[j-6]-1;r12=cl[j]/cl[max(0,j-12)]-1;r24=cl[j]/cl[max(0,j-24)]-1
  # acceleration: recent velocity minus prior-window velocity
  v3=r3/3.;v6=r6/6.;prev6=(cl[max(0,j-6)]/cl[max(0,j-12)]-1)/6. if j>=12 else 0.;acc=v6-prev6
  rng6=(hi[max(0,j-5):j+1].max()-lo[max(0,j-5):j+1].min())/cl[j];rng24=(hi[max(0,j-23):j+1].max()-lo[max(0,j-23):j+1].min())/cl[j]
  bears=[]
  for H in TF:
   bears.append(float(any((f<=ts[j]) and (f>ts[j]-pd.Timedelta(hours=max(2,2*H))) for f in d['f%d'%H])))
  # Target is not next-24h. Across the remaining regime horizon, will peak be exceeded by >=2%?
  fut=hi[j+1:n].max() if j+1<n else hi[j]; y=int(fut>=peak*1.02)
  X=[runup,dd,age/24.,j/24.,r1,r3,r6,r12,r24,v3,v6,acc,rng6,rng24]+bears+[sum(bears),bears[-1]+bears[-2]]
  ROWS.append(X+[y]);META.append((ei,j,cl[j],peak,d['mfe'],ts[j]))
A=np.asarray(ROWS,float);X=A[:,:-1];y=A[:,-1].astype(int)
print('STATES15|n=%d|events=%d|pos=%.3f|features=%d'%(len(y),len(EV),y.mean(),X.shape[1]),flush=True)
evtime=np.array([pd.Timestamp(d['ix']).value for d in EV]);order=np.argsort(evtime);cuts=np.linspace(0,len(order),6,dtype=int);OOF=np.full(len(y),np.nan)
for f in range(1,5):
 trE=set(order[:cuts[f]]);teE=set(order[cuts[f]:cuts[f+1]]);tr=np.array([m[0] in trE for m in META]);te=np.array([m[0] in teE for m in META])
 model=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,learning_rate=.06,l2_regularization=2.,min_samples_leaf=80,random_state=15)
 model.fit(X[tr],y[tr]);p=model.predict_proba(X[te])[:,1];OOF[te]=p
 print('FOLD15|%d|auc=%.3f|pmean=%.3f|n%d'%(f,roc_auc_score(y[te],p),p.mean(),te.sum()),flush=True)
ok=~np.isnan(OOF);print('OOF15|auc=%.3f|n%d'%(roc_auc_score(y[ok],OOF[ok]),ok.sum()),flush=True)

# Sell after meaningful runup when probability of another >=2% peak is low.
# Add optional trailing drawdown and persistence. Optimize capture, not just win rate.
RES=[]
for th in [.10,.15,.20,.25,.30,.35,.40]:
 for arm in [.02,.03,.05,.08]:
  for trail in [0,.015,.025,.04]:
   for persist in [1,2,3]:
    vals=[]
    for ei in order[cuts[1]:]:
     inds=[k for k,m in enumerate(META) if m[0]==ei and not np.isnan(OOF[k])]
     if not inds:continue
     d=EV[ei];e=d['e'];bad=0;chosen=None
     for k in inds:
      j=META[k][1];peak=META[k][3];runup=peak/e-1;dd=max(0,1-d['cl'][j]/peak)
      cond=runup>=arm and OOF[k]<th and dd>=trail
      bad=bad+1 if cond else 0
      if bad>=persist:chosen=k;break
     k=chosen if chosen is not None else inds[-1];j=META[k][1];px=d['cl'][j];ret=px/e-1-.003;mfe=max(0,d['mfe']);cap=ret/mfe if mfe>=.01 else np.nan;give=max(0,(META[k][3]-px)/e)
     vals.append((ret,cap,give,j/24,mfe))
    a=np.asarray(vals);caps=a[:,1];score=np.nanmedian(caps)+.5*a[:,0].mean()-.75*a[:,2].mean()+.02*(a[:,0]>0).mean()
    RES.append((score,th,arm,trail,persist,a[:,0].mean(),(a[:,0]>0).mean(),np.nanmedian(caps),a[:,2].mean(),np.median(a[:,3]),a[:,0].min(),len(a)))
for rank,r in enumerate(sorted(RES,reverse=True)[:20],1):
 print('TOP15|%d|th%.2f|arm%.2f|trail%.3f|p%d|n%d|mean%.4f|win%.3f|medcap%.3f|give%.4f|days%.2f|worst%.4f|score%.4f'%(rank,r[1],r[2],r[3],r[4],r[11],r[5],r[6],r[7],r[8],r[9],r[10],r[0]),flush=True)
# benchmarks: lower-TF SAR and fixed trailing exits
ids=order[cuts[1]:]
for H in TF:
 s=stat(('S',(H,),1,0,0),ids);print('BENCH15|SAR%dh|mean%.4f|win%.3f|medcap%.3f|give%.4f|days%.2f|worst%.4f'%(H,s[2],s[1],s[3],s[5],s[6],s[7]),flush=True)
for trl in [.02,.03,.04,.05,.06,.08,.10]:
 s=stat(('T',(trl,),1,0,0),ids);print('BENCH15|TRAIL%.2f|mean%.4f|win%.3f|medcap%.3f|give%.4f|days%.2f|worst%.4f'%(trl,s[2],s[1],s[3],s[5],s[6],s[7]),flush=True)
print('DONE15',flush=True)
