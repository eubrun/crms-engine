"""Phase10: PURE trigger = first intraday price >= fixed daily SAR. No pre-SAR geometry filters.
Goal: learn robust exit/protection + broad confirmation/regime filters without changing entry trigger.
Reuses Phase8 event construction, but excludes approach/distance features from selection.
Nested chronological selection is decisive; target >=100 trades if data supports it.
"""
exec(open('sar_phase8_presar_intraday.py').read().split("for th in [.45,.50,.55,.60]:")[0])
# At this point t contains 1325 events and OOS HGB probabilities from phase8, but those probabilities used geometry.
# Refit a CLEAN HGB excluding all geometry/distance variables; entry trigger remains unconditional SAR cross.
geom={'gap_to_sar','d1','d3','d5','d7','approach3','approach5','approach7','dist_slope3','dist_slope5','dist_slope7','conv3','conv5','sar_slope3'}
base_ex={'symbol','i','trigger','entry','_path','label','prob'}|geom
F=[c for c in t.columns if c not in base_ex]
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
idx=pd.DatetimeIndex(pd.to_datetime(t.index,utc=True)).tz_convert(None);D=idx.unique().sort_values();bounds=[.50,.60,.70,.80,.90,1.0];P=np.full(len(t),np.nan);X=t[F].astype(float);y=t.label.to_numpy()
for f in range(5):
 a=D[int(len(D)*bounds[f])];b=D[-1]+pd.Timedelta(days=1) if f==4 else D[int(len(D)*bounds[f+1])];tr=idx<a;te=(idx>=a)&(idx<b);m=make_pipeline(SimpleImputer(),HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,learning_rate=.04,l2_regularization=4,min_samples_leaf=25,random_state=10));m.fit(X.loc[tr],y[tr]);P[te]=m.predict_proba(X.loc[te])[:,1];print('FOLD10|%d|train=%d|test=%d'%(f+1,tr.sum(),te.sum()),flush=True)
t['pclean']=P
# exits: fixed TP/SL/time plus trailing. Hourly conservative ambiguity => adverse exit.
def fixed(r,tp,sl,H):
 e=float(r.entry);up=e*(1+tp);dn=e*(1-sl);p=r['_path'];end=pd.Timestamp(r.name)+pd.Timedelta(days=H+1);p=p[p.index<end]
 for _,b in p.iterrows():
  hu=b.high>=up;hd=b.low<=dn
  if hu and hd:return -sl-.003
  if hd:return -sl-.003
  if hu:return tp-.003
 return (float(p.close.iloc[-1]/e-1)-.003) if len(p) else np.nan
def trail(r,arm,tr,H):
 e=float(r.entry);p=r['_path'];end=pd.Timestamp(r.name)+pd.Timedelta(days=H+1);p=p[p.index<end];peak=e;armed=False
 for _,b in p.iterrows():
  peak=max(peak,float(b.high));armed=armed or peak>=e*(1+arm)
  if armed:
   stop=peak*(1-tr)
   if float(b.low)<=stop:return stop/e-1-.003
 return (float(p.close.iloc[-1]/e-1)-.003) if len(p) else np.nan
cfg=[]
for tp in [.08,.10,.12,.15,.18]:
 for sl in [.04,.05,.06,.075,.10]:
  for H in [3,5,7,10,12,15]:cfg.append(('F',tp,sl,H))
for arm in [.05,.08,.10]:
 for tr in [.04,.06,.08]:
  for H in [5,7,10,12,15]:cfg.append(('T',arm,tr,H))
def run(r,c):return fixed(r,c[1],c[2],c[3]) if c[0]=='F' else trail(r,c[1],c[2],c[3])
def st(g,c):
 rr=np.array([run(r,c) for _,r in g.iterrows()]);rr=rr[np.isfinite(rr)];return len(rr),(rr>0).mean(),rr.mean(),np.median(rr),rr.min()
# broad clean probability thresholds; no geometry
for th in [.30,.35,.40,.45,.50,.55]:
 g=t[t.pclean>=th];best=[]
 for c in cfg:
  n,w,me,md,wo=st(g,c);score=me+.03*w+.15*wo;best.append((score,c,n,w,me,md,wo))
 q=sorted(best,reverse=True)[:6]
 print('COUNT10|th=%.2f|n=%d'%(th,len(g)),flush=True)
 for j,(sc,c,n,w,me,md,wo) in enumerate(q,1):print('TOP10|th=%.2f|rank=%d|cfg=%s/%.3f/%.3f/%d|n=%d|win=%.3f|mean=%.4f|med=%.4f|worst=%.4f|score=%.4f'%(th,j,c[0],c[1],c[2],c[3],n,w,me,md,wo,sc),flush=True)
# Nested selection using prior OOS only. Candidate threshold+exit must preserve at least 35% of prior OOS events to prevent sample collapse.
ALL=[]
for f in range(1,5):
 h0=D[int(len(D)*.50)];h1=D[int(len(D)*bounds[f])];q0=h1;q1=D[-1]+pd.Timedelta(days=1) if f==4 else D[int(len(D)*bounds[f+1])];hist=t[(idx>=h0)&(idx<h1)];test=t[(idx>=q0)&(idx<q1)];best=None
 for th in [.30,.35,.40,.45,.50,.55]:
  hg=hist[hist.pclean>=th]
  if len(hg)<max(25,int(.35*len(hist))):continue
  for c in cfg:
   n,w,me,md,wo=st(hg,c);score=me+.03*w+.15*wo+.002*min(n,100)/100
   if best is None or score>best[0]:best=(score,th,c,n)
 if best is None:continue
 _,th,c,hn=best;g=test[test.pclean>=th];n,w,me,md,wo=st(g,c);rr=[run(r,c) for _,r in g.iterrows()];ALL += [x for x in rr if np.isfinite(x)];print('NEST10|fold=%d|th=%.2f|cfg=%s/%.3f/%.3f/%d|histn=%d|n=%d|win=%.3f|mean=%.4f|worst=%.4f'%(f+1,th,c[0],c[1],c[2],c[3],hn,n,w,me,wo),flush=True)
if ALL:
 a=np.array(ALL);print('NESTSUM10|n=%d|win=%.3f|mean=%.4f|median=%.4f|worst=%.4f'%(len(a),(a>0).mean(),a.mean(),np.median(a),a.min()),flush=True)
# unconditional trigger benchmark: every SAR cross, robust exits ranked
best=[]
for c in cfg:
 n,w,me,md,wo=st(t,c);score=me+.03*w+.15*wo;best.append((score,c,n,w,me,md,wo))
for j,(sc,c,n,w,me,md,wo) in enumerate(sorted(best,reverse=True)[:10],1):print('ALL10|rank=%d|cfg=%s/%.3f/%.3f/%d|n=%d|win=%.3f|mean=%.4f|med=%.4f|worst=%.4f'%(j,c[0],c[1],c[2],c[3],n,w,me,md,wo),flush=True)
print('DONE10',flush=True)
