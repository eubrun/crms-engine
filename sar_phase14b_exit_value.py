"""Phase14B: causal exit-value study after Daily-SAR BUY.
Build EV directly from Phase13B preparation, bypassing Phase14 classifier/RULE14.
Chronological event-level OOF only.
"""
# Execute only Phase13B preparation through EV/MFE creation; do not run its exit grid.
exec(open('sar_phase13b_exit_fast.py').read().split("# 90+ meaningful configurations")[0])
from sklearn.ensemble import HistGradientBoostingRegressor

# Causal hourly states. Features use information known at hour j only.
ROWS=[]; META=[]; YU=[]; YD=[]
for ei,d in enumerate(EV):
 e=d['e']; hi=d['hi']; lo=d['lo']; cl=d['cl']; ts=d['ts']; n=min(len(ts),240)
 peak=e; peakj=0
 for j in range(2,n-1):
  if hi[j]>=peak: peak=hi[j]; peakj=j
  runup=peak/e-1; dd=cl[j]/peak-1; age=(j-peakj)/24; life=j/24
  r1=cl[j]/cl[j-1]-1; r2=cl[j]/cl[max(0,j-2)]-1; r6=cl[j]/cl[max(0,j-6)]-1
  r12=cl[j]/cl[max(0,j-12)]-1; r24=cl[j]/cl[max(0,j-24)]-1
  rng=(hi[max(0,j-23):j+1].max()-lo[max(0,j-23):j+1].min())/cl[j]
  bear=[]
  for H in TF:
   bear.append(float(any((f<=ts[j]) and (f>ts[j]-pd.Timedelta(hours=max(2,H))) for f in d['f%d'%H])))
  X=[runup,dd,age,life,r1,r2,r6,r12,r24,rng]+bear+[sum(bear)]
  end=min(n,j+25)
  fh=hi[j+1:end].max() if end>j+1 else cl[j]
  fl=lo[j+1:end].min() if end>j+1 else cl[j]
  # continuation above current running peak vs downside from current close over next 24h
  yu=max(0.0,fh/peak-1.0); yd=max(0.0,1.0-fl/cl[j])
  ROWS.append(X); META.append((ei,j,cl[j],peak,d['mfe'],ts[j])); YU.append(yu); YD.append(yd)
X=np.asarray(ROWS,float);YU=np.asarray(YU);YD=np.asarray(YD)
print('STATES14B|n=%d|events=%d|features=%d|upmean=%.4f|downmean=%.4f'%(len(X),len(EV),X.shape[1],YU.mean(),YD.mean()),flush=True)

evtime=np.array([pd.Timestamp(d['ix']).value for d in EV]); order=np.argsort(evtime); cuts=np.linspace(0,len(order),6,dtype=int)
OOFU=np.full(len(YU),np.nan);OOFD=np.full(len(YD),np.nan)
for f in range(1,5):
 train_ev=set(order[:cuts[f]]); test_ev=set(order[cuts[f]:cuts[f+1]])
 tr=np.array([m[0] in train_ev for m in META]); te=np.array([m[0] in test_ev for m in META])
 kw=dict(max_iter=160,max_leaf_nodes=15,learning_rate=.06,l2_regularization=2.0,min_samples_leaf=80,random_state=141)
 mu=HistGradientBoostingRegressor(**kw);md=HistGradientBoostingRegressor(**kw)
 mu.fit(X[tr],YU[tr]);md.fit(X[tr],YD[tr])
 OOFU[te]=np.maximum(0,mu.predict(X[te]));OOFD[te]=np.maximum(0,md.predict(X[te]))
 print('FOLD14B|%d|train=%d|test=%d|upMAE=%.4f|downMAE=%.4f'%(f,tr.sum(),te.sum(),np.mean(np.abs(OOFU[te]-YU[te])),np.mean(np.abs(OOFD[te]-YD[te]))),flush=True)

# SELL when expected continuation is dominated by lambda-weighted downside.
# Arm only after realized run-up and require persistence to suppress hourly noise.
RESULT=[]
for lam in [.75,1.,1.25,1.5,2.]:
 for arm in [.01,.02,.03,.05]:
  for persist in [1,2,3,4]:
   vals=[]
   for ei in order[cuts[1]:]:
    inds=[k for k,m in enumerate(META) if m[0]==ei and not np.isnan(OOFU[k])]
    if not inds: continue
    d=EV[ei];e=d['e'];chosen=None;bad=0
    for k in inds:
     runup=META[k][3]/e-1; score=OOFU[k]-lam*OOFD[k]
     bad=bad+1 if (runup>=arm and score<0) else 0
     if bad>=persist: chosen=k;break
    k=chosen if chosen is not None else inds[-1]
    j=META[k][1];px=d['cl'][j];ret=px/e-1-.003;mfe=max(0,d['mfe'])
    cap=ret/mfe if mfe>=.01 else np.nan;give=max(0,(META[k][3]-px)/e)
    vals.append((ret,cap,give,j/24,mfe))
   a=np.asarray(vals);caps=a[:,1]
   row=(a[:,0].mean(),(a[:,0]>0).mean(),np.nanmedian(caps),np.nanmean(caps),a[:,2].mean(),np.median(a[:,3]),a[:,0].min(),lam,arm,persist,len(a))
   RESULT.append(row)
   print('RULE14B|L%.2f|arm%.2f|p%d|n%d|win%.3f|mean%.4f|medcap%.3f|avgcap%.3f|give%.4f|days%.2f|worst%.4f'%(lam,arm,persist,len(a),row[1],row[0],row[2],row[3],row[4],row[5],row[6]),flush=True)
for rank,r in enumerate(sorted(RESULT,reverse=True,key=lambda z:z[0])[:10],1):
 print('TOP14B|%d|L%.2f|arm%.2f|p%d|mean%.4f|win%.3f|medcap%.3f|give%.4f|days%.2f|worst%.4f'%(rank,r[7],r[8],r[9],r[0],r[1],r[2],r[4],r[5],r[6]),flush=True)
print('DONE14B',flush=True)
