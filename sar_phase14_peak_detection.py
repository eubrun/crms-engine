"""Phase14 causal peak-detection study.
BUY is fixed at intraday Daily-SAR cross. Learn when continuation probability collapses.
Uses only information available at each hour; chronological out-of-sample validation.
"""
exec(open('sar_phase13b_exit_fast.py').read().split("# 90+ meaningful configurations")[0])
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# Build hourly causal states. Target: after this hour, will price make a meaningfully new high
# (+1% over current running peak) within next 48h before suffering 5% drawdown from that peak?
ROWS=[]; META=[]
for ei,d in enumerate(EV):
 e=d['e']; hi=d['hi']; lo=d['lo']; cl=d['cl']; ts=d['ts']; n=min(len(ts),240)
 peak=e; peakj=0
 # bearish SAR state inferred causally from completed flip timestamps: once flipped, mark a short window;
 # distances/price dynamics and drawdown carry most of the continuous information.
 flips={H:set(d['f%d'%H]) for H in TF}
 for j in range(2,n-1):
  if hi[j]>=peak: peak=hi[j];peakj=j
  runup=peak/e-1; dd=cl[j]/peak-1; age=j-peakj
  r1=cl[j]/cl[j-1]-1; r2=cl[j]/cl[max(0,j-2)]-1; r6=cl[j]/cl[max(0,j-6)]-1; r12=cl[j]/cl[max(0,j-12)]-1; r24=cl[j]/cl[max(0,j-24)]-1
  rng=(hi[max(0,j-23):j+1].max()-lo[max(0,j-23):j+1].min())/cl[j]
  bear=[]
  for H in TF:
   # most recent completed flip bearish remains a deterioration marker for H hours
   bear.append(float(any((f<=ts[j]) and (f>ts[j]-pd.Timedelta(hours=max(2,H))) for f in d['f%d'%H])))
  end=min(n,j+49); fut_hi=hi[j+1:end].max() if end>j+1 else hi[j]; fut_lo=lo[j+1:end].min() if end>j+1 else lo[j]
  newhigh=fut_hi>=peak*1.01
  risk=fut_lo<=peak*.95
  y=int(newhigh and not risk)
  X=[runup,dd,age/24,j/24,r1,r2,r6,r12,r24,rng]+bear+[sum(bear)]
  ROWS.append(X+[y]); META.append((ei,j,cl[j],peak,d['mfe'],ts[j]))
A=np.asarray(ROWS,float); X=A[:,:-1]; y=A[:,-1].astype(int)
print('STATES14|n=%d|events=%d|positive=%.3f|features=%d'%(len(y),len(EV),y.mean(),X.shape[1]),flush=True)

# Event-level chronological folds prevent states from same event leaking across train/test.
evtime=np.array([pd.Timestamp(d['ix']).value for d in EV]); order=np.argsort(evtime); cuts=np.linspace(0,len(order),6,dtype=int)
OOF=np.full(len(y),np.nan)
for f in range(1,5):
 train_ev=set(order[:cuts[f]]); test_ev=set(order[cuts[f]:cuts[f+1]])
 tr=np.array([m[0] in train_ev for m in META]); te=np.array([m[0] in test_ev for m in META])
 model=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,learning_rate=.06,l2_regularization=2.0,min_samples_leaf=80,random_state=14)
 model.fit(X[tr],y[tr]); p=model.predict_proba(X[te])[:,1];OOF[te]=p
 auc=roc_auc_score(y[te],p) if len(np.unique(y[te]))>1 else np.nan
 print('FOLD14|%d|trainstates=%d|teststates=%d|auc=%.3f|pmean=%.3f'%(f,len(y[tr]),len(y[te]),auc,p.mean()),flush=True)

ok=~np.isnan(OOF);print('OOF14|states=%d|auc=%.3f'%(ok.sum(),roc_auc_score(y[ok],OOF[ok])),flush=True)
# Simulate causal exits: sell when continuation probability falls below threshold,
# but only after minimum run-up; compare capture/giveback/return.
for th in [.15,.20,.25,.30,.35,.40,.45,.50]:
 for arm in [0,.02,.05,.08,.10]:
  vals=[]
  for ei in order[cuts[1]:]:
   inds=[k for k,m in enumerate(META) if m[0]==ei and not np.isnan(OOF[k])]
   if not inds:continue
   d=EV[ei];e=d['e'];chosen=None
   for k in inds:
    j=META[k][1];peak=META[k][3]
    if peak/e-1>=arm and OOF[k]<th:chosen=k;break
   if chosen is None:k=inds[-1]
   else:k=chosen
   j=META[k][1];px=d['cl'][j];ret=px/e-1-.003;mfe=max(d['mfe'],1e-9);cap=ret/mfe;give=max(0,(META[k][3]-px)/e)
   vals.append((ret,cap,give,j/24))
  a=np.asarray(vals);print('RULE14|th%.2f|arm%.2f|n%d|win%.3f|mean%.4f|medcap%.3f|avgcap%.3f|give%.4f|days%.2f|worst%.4f'%(th,arm,len(a),(a[:,0]>0).mean(),a[:,0].mean(),np.median(a[:,1]),np.mean(a[:,1]),a[:,2].mean(),np.median(a[:,3]),a[:,0].min()),flush=True)
print('DONE14',flush=True)
