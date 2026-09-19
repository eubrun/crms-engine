"""Phase12: FIXED entry trigger = first intraday price >= daily SAR. No pre-SAR geometry.
Classify cross quality using only information available at trigger/prior completed daily bar.
Study stability by market regime and chronological OOS. Exit fixed TP18/SL6/max5d so only entry-quality is tested.
"""
exec(open('sar_phase8_presar_intraday.py').read().split("for th in [.45,.50,.55,.60]:")[0])
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
COST=.003
# outcome under fixed execution; this is what quality model predicts
def outcome(r):
 e=float(r.entry);up=e*1.18;dn=e*.94;p=r['_path'];end=pd.Timestamp(r.name)+pd.Timedelta(days=6);p=p[p.index<end]
 for _,b in p.iterrows():
  if b.low<=dn:return -.063
  if b.high>=up:return .177
 return float(p.close.iloc[-1]/e-1)-COST if len(p) else np.nan
t['ret_exec']=t.apply(outcome,axis=1);t['ywin']=(t.ret_exec>0).astype(int)
# EXCLUDE every geometry/pre-SAR feature and phase8 label/prob/path metadata.
geom={'gap_to_sar','d1','d3','d5','d7','approach3','approach5','approach7','dist_slope3','dist_slope5','dist_slope7','conv3','conv5','sar_slope3'}
meta={'symbol','i','trigger','entry','_path','label','prob','ret_exec','ywin'}
F=[c for c in t.columns if c not in geom|meta]
print('FEATURES12|'+','.join(F),flush=True)
# regime labels from BTC features already shifted to prior completed daily bar
# 3 regimes: bull if btc20>0 & above50 & sarbull; bear if btc20<0 & not above50; else neutral
t['regime']=np.where((t.btc20>0)&(t.btc_above50==1)&(t.btc_sarbull==1),'BULL',np.where((t.btc20<0)&(t.btc_above50==0),'BEAR','NEUTRAL'))
for rg,g in t.groupby('regime'):
 print('REGIME12|%s|n=%d|win=%.3f|mean=%.4f'%(rg,len(g),g.ywin.mean(),g.ret_exec.mean()),flush=True)
X=t[F].astype(float);y=t.ywin.to_numpy();idx=pd.DatetimeIndex(pd.to_datetime(t.index,utc=True)).tz_convert(None);D=idx.unique().sort_values();bounds=[.50,.60,.70,.80,.90,1.0];P=np.full(len(t),np.nan)
# chronological HGB; no geometry
for f in range(5):
 a=D[int(len(D)*bounds[f])];b=D[-1]+pd.Timedelta(days=1) if f==4 else D[int(len(D)*bounds[f+1])];tr=idx<a;te=(idx>=a)&(idx<b)
 m=make_pipeline(SimpleImputer(),HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=7,learning_rate=.035,l2_regularization=6,min_samples_leaf=30,random_state=12));m.fit(X.loc[tr],y[tr]);P[te]=m.predict_proba(X.loc[te])[:,1]
 print('FOLD12|%d|train=%d|test=%d|basewin=%.3f'%(f+1,tr.sum(),te.sum(),y[te].mean()),flush=True)
t['p12']=P
# threshold performance OOS. Need both trade count and fold stability.
for th in [.30,.35,.40,.45,.50,.55,.60]:
 g=t[t.p12>=th];foldwins=[]
 for f in range(5):
  a=D[int(len(D)*bounds[f])];b=D[-1]+pd.Timedelta(days=1) if f==4 else D[int(len(D)*bounds[f+1])];q=g[(pd.DatetimeIndex(pd.to_datetime(g.index,utc=True)).tz_convert(None)>=a)&(pd.DatetimeIndex(pd.to_datetime(g.index,utc=True)).tz_convert(None)<b)]
  if len(q):foldwins.append(q.ywin.mean())
 print('TH12|th=%.2f|n=%d|win=%.3f|mean=%.4f|worstfold=%.3f|folds=%d'%(th,len(g),g.ywin.mean() if len(g) else np.nan,g.ret_exec.mean() if len(g) else np.nan,min(foldwins) if foldwins else np.nan,len(foldwins)),flush=True)
# regime-conditioned thresholds, descriptive OOS
for rg in ['BULL','NEUTRAL','BEAR']:
 for th in [.35,.40,.45,.50,.55]:
  g=t[(t.regime==rg)&(t.p12>=th)]
  if len(g)>=15:print('RTH12|%s|th=%.2f|n=%d|win=%.3f|mean=%.4f'%(rg,th,len(g),g.ywin.mean(),g.ret_exec.mean()),flush=True)
# nested threshold selection from prior OOS, constrained >=35% historical OOS events and objective expectancy + win + fold robustness
ALL=[]
for f in range(1,5):
 h0=D[int(len(D)*.50)];h1=D[int(len(D)*bounds[f])];q0=h1;q1=D[-1]+pd.Timedelta(days=1) if f==4 else D[int(len(D)*bounds[f+1])];hist=t[(idx>=h0)&(idx<h1)];test=t[(idx>=q0)&(idx<q1)];best=None
 for th in [.30,.35,.40,.45,.50,.55]:
  g=hist[hist.p12>=th]
  if len(g)<max(25,int(.35*len(hist))):continue
  score=g.ret_exec.mean()+.03*g.ywin.mean()+.002*min(len(g),100)/100
  if best is None or score>best[0]:best=(score,th,len(g))
 th=best[1];g=test[test.p12>=th];ALL.append(g)
 print('NEST12|fold=%d|th=%.2f|histn=%d|n=%d|win=%.3f|mean=%.4f'%(f+1,th,best[2],len(g),g.ywin.mean() if len(g) else np.nan,g.ret_exec.mean() if len(g) else np.nan),flush=True)
a=pd.concat(ALL);print('NESTSUM12|n=%d|win=%.3f|mean=%.4f|median=%.4f|worst=%.4f'%(len(a),a.ywin.mean(),a.ret_exec.mean(),a.ret_exec.median(),a.ret_exec.min()),flush=True)
# Univariate diagnostics by quartile for interpretable cross characteristics
for c in F:
 try:
  q=pd.qcut(t[c],4,duplicates='drop');vals=[]
  for k,g in t.groupby(q,observed=True):vals.append('%d:%.3f:%.4f'%(len(g),g.ywin.mean(),g.ret_exec.mean()))
  print('UNI12|%s|%s'%(c,';'.join(vals)),flush=True)
 except:pass
print('DONE12',flush=True)
