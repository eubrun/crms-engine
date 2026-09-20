"""Phase14B: causal exit-value study after Daily-SAR BUY.
Sell when expected near-term giveback risk dominates continuation potential.
Chronological event-level OOF only; no future information enters features.
"""
exec(open('sar_phase14_peak_detection.py').read().split("# Simulate causal exits:")[0])
from sklearn.ensemble import HistGradientBoostingRegressor

# Rebuild causal targets for each state: next-24h upside from current running peak
# versus next-24h giveback below current close. Train two regressors separately.
YU=[]; YD=[]
for m in META:
 ei,j,px,peak,mfe,t=m; d=EV[ei]; n=min(len(d['ts']),240); end=min(n,j+25)
 fh=d['hi'][j+1:end].max() if end>j+1 else px
 fl=d['lo'][j+1:end].min() if end>j+1 else px
 YU.append(max(0.0,fh/peak-1.0))
 YD.append(max(0.0,1.0-fl/px))
YU=np.asarray(YU);YD=np.asarray(YD)
OOFU=np.full(len(YU),np.nan);OOFD=np.full(len(YD),np.nan)
for f in range(1,5):
 train_ev=set(order[:cuts[f]]); test_ev=set(order[cuts[f]:cuts[f+1]])
 tr=np.array([m[0] in train_ev for m in META]); te=np.array([m[0] in test_ev for m in META])
 kw=dict(max_iter=160,max_leaf_nodes=15,learning_rate=.06,l2_regularization=2.0,min_samples_leaf=80,random_state=141)
 mu=HistGradientBoostingRegressor(**kw); md=HistGradientBoostingRegressor(**kw)
 mu.fit(X[tr],YU[tr]);md.fit(X[tr],YD[tr]);OOFU[te]=np.maximum(0,mu.predict(X[te]));OOFD[te]=np.maximum(0,md.predict(X[te]))
 print('FOLD14B|%d|upMAE=%.4f|downMAE=%.4f'%(f,np.mean(np.abs(OOFU[te]-YU[te])),np.mean(np.abs(OOFD[te]-YD[te]))),flush=True)

# value = expected upside minus lambda * expected giveback.
# Arm only after a modest realized runup; require deterioration persistence to avoid one-hour noise.
for lam in [0.75,1.0,1.25,1.5,2.0]:
 for arm in [.01,.02,.03,.05]:
  for persist in [1,2,3,4]:
   vals=[]
   for ei in order[cuts[1]:]:
    inds=[k for k,m in enumerate(META) if m[0]==ei and not np.isnan(OOFU[k])]
    if not inds: continue
    d=EV[ei];e=d['e'];chosen=None;bad=0
    for k in inds:
     j=META[k][1];peak=META[k][3];runup=peak/e-1
     score=OOFU[k]-lam*OOFD[k]
     if runup>=arm and score<0: bad+=1
     else: bad=0
     if bad>=persist: chosen=k;break
    k=chosen if chosen is not None else inds[-1]
    j=META[k][1];px=d['cl'][j];ret=px/e-1-.003;mfe=max(0,d['mfe']);
    cap=ret/mfe if mfe>=.01 else np.nan
    give=max(0,(META[k][3]-px)/e)
    vals.append((ret,cap,give,j/24,mfe))
   a=np.asarray(vals);caps=a[:,1];valid=~np.isnan(caps)
   print('RULE14B|L%.2f|arm%.2f|p%d|n%d|win%.3f|mean%.4f|med%.4f|medcap%.3f|avgcap%.3f|give%.4f|days%.2f|worst%.4f'%(lam,arm,persist,len(a),(a[:,0]>0).mean(),a[:,0].mean(),np.median(a[:,0]),np.nanmedian(caps),np.nanmean(caps),a[:,2].mean(),np.median(a[:,3]),a[:,0].min()),flush=True)
print('DONE14B',flush=True)
