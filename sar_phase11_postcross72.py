"""Phase11: entry trigger is FIXED: first intraday price >= daily SAR. No pre-SAR geometry.
Study only post-entry information at 12/24/48/72h to cut failed crosses early.
Baseline terminal exit fixed at TP18%, SL6%, max 5d. Early-exit rules use only information observable by checkpoint.
Chronological nested selection is decisive; entry count is unchanged, only exits adapt.
"""
exec(open('sar_phase8_presar_intraday.py').read().split("for th in [.45,.50,.55,.60]:")[0])
# Do NOT use Phase8 probability/geometry. Every event is entered.
COST=.003

def baseline(r):
 e=float(r.entry);up=e*1.18;dn=e*.94;p=r['_path'];end=pd.Timestamp(r.name)+pd.Timedelta(days=6);p=p[p.index<end]
 for _,b in p.iterrows():
  if b.low<=dn:return -.06-COST
  if b.high>=up:return .18-COST
 return float(p.close.iloc[-1]/e-1)-COST if len(p) else np.nan
# checkpoint metrics from hourly bars AFTER entry; all causal.
def checkpoint(r,hours):
 e=float(r.entry);p=r['_path'].iloc[:hours]
 if len(p)<max(6,hours//2):return None
 ret=float(p.close.iloc[-1]/e-1);mfe=float(p.high.max()/e-1);mae=float(p.low.min()/e-1)
 # last 6h momentum, volume ratio first/last halves, drawdown from peak
 mom6=float(p.close.iloc[-1]/p.close.iloc[max(0,len(p)-7)]-1) if len(p)>6 else ret
 half=max(1,len(p)//2);v1=float(p.volume.iloc[:half].mean());v2=float(p.volume.iloc[half:].mean());vr=v2/v1 if v1>0 else 1
 dd=float(p.close.iloc[-1]/p.high.max()-1)
 return ret,mfe,mae,mom6,vr,dd,float(p.close.iloc[-1]/e-1)-COST
# Early rules intentionally broad/interpretable: if by checkpoint insufficient progress OR deteriorating from peak, exit.
# tuple (hours,min_mfe,min_ret,min_mom6,max_dd). None disables criterion.
rules=[]
for H in [12,24,48,72]:
 for mfe in [0,.01,.02,.03,.05]:
  for ret in [-.04,-.02,0,.01,.02]:
   for mom in [-.03,-.01,0]:
    rules.append((H,mfe,ret,mom,None))
# separate peak-failure family
for H in [24,48,72]:
 for mfe in [.02,.04,.06]:
  for dd in [-.06,-.04,-.03,-.02]:rules.append((H,mfe,-.99,-.99,dd))

def execute(r,rule):
 e=float(r.entry);p=r['_path'];up=e*1.18;dn=e*.94;H,mfe0,ret0,mom0,dd0=rule
 # process until checkpoint, honoring hard TP/SL first
 pre=p.iloc[:H]
 for _,b in pre.iterrows():
  if b.low<=dn:return -.06-COST,'SL'
  if b.high>=up:return .18-COST,'TP'
 cp=checkpoint(r,H)
 if cp is None:return baseline(r),'BASE'
 ret,mfe,mae,mom,vr,dd,exitret=cp
 fail=(mfe<mfe0) or (ret<ret0) or (mom<mom0)
 if dd0 is not None:fail=(mfe>=mfe0 and dd<dd0)
 if fail:return exitret,'EARLY'
 # continue from checkpoint to day5
 end=pd.Timestamp(r.name)+pd.Timedelta(days=6);post=p.iloc[H:];post=post[post.index<end]
 for _,b in post.iterrows():
  if b.low<=dn:return -.06-COST,'SL'
  if b.high>=up:return .18-COST,'TP'
 q=p[p.index<end];return (float(q.close.iloc[-1]/e-1)-COST if len(q) else np.nan),'TIME'

def stats(g,rule):
 o=[execute(r,rule) for _,r in g.iterrows()];x=np.array([a for a,_ in o if np.isfinite(a)]);early=sum(k=='EARLY' for _,k in o);return len(x),(x>0).mean(),x.mean(),np.median(x),x.min(),early
# Base benchmark all 1325
b=np.array([baseline(r) for _,r in t.iterrows()]);print('BASE11|n=%d|win=%.3f|mean=%.4f|med=%.4f|worst=%.4f'%(len(b),(b>0).mean(),b.mean(),np.median(b),b.min()),flush=True)
# descriptive full-history ranking, but NOT used as proof
Q=[]
for r in rules:
 n,w,me,md,wo,early=stats(t,r);score=me+.03*w+.15*wo;Q.append((score,r,n,w,me,md,wo,early))
for j,(sc,r,n,w,me,md,wo,early) in enumerate(sorted(Q,reverse=True)[:15],1):print('TOP11|rank=%d|H=%d|mfe=%.3f|ret=%.3f|mom=%.3f|dd=%s|n=%d|win=%.3f|mean=%.4f|med=%.4f|worst=%.4f|early=%d'%(j,r[0],r[1],r[2],r[3],str(r[4]),n,w,me,md,wo,early),flush=True)
# nested chronological: each future fold chooses rule using only prior OOS periods. Keep all entries; only exit differs.
idx=pd.DatetimeIndex(pd.to_datetime(t.index,utc=True)).tz_convert(None);D=idx.unique().sort_values();bounds=[.50,.60,.70,.80,.90,1.0];ALL=[]
for f in range(1,5):
 h0=D[int(len(D)*.50)];h1=D[int(len(D)*bounds[f])];q0=h1;q1=D[-1]+pd.Timedelta(days=1) if f==4 else D[int(len(D)*bounds[f+1])];hist=t[(idx>=h0)&(idx<h1)];test=t[(idx>=q0)&(idx<q1)];best=None
 for r in rules:
  n,w,me,md,wo,early=stats(hist,r);score=me+.03*w+.15*wo
  if best is None or score>best[0]:best=(score,r)
 r=best[1];o=[execute(x,r)[0] for _,x in test.iterrows()];x=np.array(o);ALL+=o;n,w,me,md,wo,early=stats(test,r);print('NEST11|fold=%d|H=%d|mfe=%.3f|ret=%.3f|mom=%.3f|dd=%s|n=%d|win=%.3f|mean=%.4f|worst=%.4f|early=%d'%(f+1,r[0],r[1],r[2],r[3],str(r[4]),n,w,me,wo,early),flush=True)
a=np.array(ALL);print('NESTSUM11|n=%d|win=%.3f|mean=%.4f|median=%.4f|worst=%.4f'%(len(a),(a>0).mean(),a.mean(),np.median(a),a.min()),flush=True)
# simple fixed rules chosen a priori for robustness comparison
for r in [(24,.01,-.02,-.01,None),(24,.02,-.02,-.01,None),(48,.02,-.02,-.01,None),(48,.03,0,-.01,None),(72,.03,0,-.01,None)]:
 n,w,me,md,wo,e=stats(t,r);print('FIX11|H=%d|mfe=%.3f|ret=%.3f|mom=%.3f|n=%d|win=%.3f|mean=%.4f|worst=%.4f|early=%d'%(r[0],r[1],r[2],r[3],n,w,me,wo,e),flush=True)
print('DONE11',flush=True)
