"""Phase24: ML gate at first 8H bearish PSAR event. Direct counterfactual reward: SELL now vs HOLD to next 8H bearish episode. Entry fixed Daily PSAR cross."""
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
exec(open('sar_phase18c_optimizer.py').read().split("# Candidate rule descriptors")[0])
N=len(O);TFS=[2,4,6,8,12,24];print('OOS24|n=%d'%N,flush=True)
H={}
for sym in sorted(set(x['sym'] for x in O)):
 xs=[x for x in O if x['sym']==sym];st=min(x['ts'] for x in xs)-pd.Timedelta(days=5);en=max(x['ts'] for x in xs)+pd.Timedelta(hours=1081)
 try:H[sym]=hourly(sym,int(st.timestamp()*1000),int(en.timestamp()*1000));print('H24|%s|%d'%(sym,len(H[sym])),flush=True)
 except Exception as e:H[sym]=pd.DataFrame();print('FAIL24|%s|%s'%(sym,str(e)[:100]),flush=True)
def atr(df,n=24):
 tr=pd.concat([df.high-df.low,(df.high-df.close.shift()).abs(),(df.low-df.close.shift()).abs()],axis=1).max(axis=1);return tr.rolling(n,min_periods=5).mean()
def adx(df,n=14):
 h,l,c=df.high,df.low,df.close;up=h.diff();dn=-l.diff();tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1);a=tr.ewm(alpha=1/n,adjust=False).mean();pd_=up.where((up>dn)&(up>0),0).ewm(alpha=1/n,adjust=False).mean();md_=dn.where((dn>up)&(dn>0),0).ewm(alpha=1/n,adjust=False).mean();p=100*pd_/a.replace(0,np.nan);m=100*md_/a.replace(0,np.nan);dx=100*(p-m).abs()/(p+m).replace(0,np.nan);return dx.ewm(alpha=1/n,adjust=False).mean(),p,m
rows=[];paths={}
for ci,x in enumerate(O):
 p=H.get(x['sym'],pd.DataFrame());p=p[(p.index>=x['ts'])&(p.index<x['ts']+pd.Timedelta(hours=1081))]
 if p.empty or len(p)<48:continue
 entry=float(x['entry']);states={}
 for tf in TFS:
  z=bars(p,tf);bb,ss=psar_state(z);states[tf]=(z,bb,ss)
 z8,b8,s8=states[8]; flips=np.where((b8[1:]!=b8[:-1]) & (~b8[1:]))[0]+1
 if not len(flips):continue
 # map bearish 8H episodes to first available hourly close at/after completed 8H bar timestamp
 ev=[]
 for k in flips:
  j=p.index.searchsorted(z8.index[k],'left');
  if j<len(p):ev.append(j)
 if not ev:continue
 paths[ci]=(p,entry,ev)
 for ei,j in enumerate(ev):
  t=p.index[j];px=float(p.close.iloc[j]);runhi=float(p.high.iloc[:j+1].max());ret=px/entry-1;mfe=runhi/entry-1;give=px/runhi-1;age=j/24
  # next eligible 8H bearish episode, otherwise horizon end
  j2=ev[ei+1] if ei+1<len(ev) else len(p)-1;holdret=float(p.close.iloc[j2]/entry-1);delta=holdret-ret
  r=p.close.pct_change();feat=[ret,mfe,give,age,float(r.iloc[max(0,j-23):j+1].std()),float(r.iloc[max(0,j-71):j+1].std()),float(px/p.close.iloc[max(0,j-6)]-1),float(px/p.close.iloc[max(0,j-12)]-1),float(px/p.close.iloc[max(0,j-24)]-1),float(px/p.close.iloc[max(0,j-72)]-1)]
  bears=[]
  for tf in TFS:
   z,b,s=states[tf];k=z.index.searchsorted(t,'right')-1
   if k<0:feat += [0,0];bears.append(0)
   else:
    be=float(not b[k]);feat += [be,(px-float(s[k]))/px if np.isfinite(s[k]) else 0];bears.append(be)
  feat += [sum(bears),sum((i+1)*v for i,v in enumerate(bears))/21]
  # 8H trend strength
  q=bars(p.iloc[:j+1],8);A=atr(q,14);X,P,M=adx(q,14);e20=q.close.ewm(span=20,adjust=False).mean();e50=q.close.ewm(span=50,adjust=False).mean();k=len(q)-1
  def vv(s):return float(s.iloc[k]) if k>=0 and np.isfinite(s.iloc[k]) else 0
  feat += [vv(X),vv(P)-vv(M),vv(A)/px if px else 0,(px-vv(e20))/px,(px-vv(e50))/px]
  rows.append([ci,ei,t,ret,holdret,delta,j,j2]+feat)
base=['ret','mfe','give','age','vol24','vol72','mom6','mom12','mom24','mom72'];sf=[]
for tf in TFS:sf += ['bear%d'%tf,'psard%d'%tf]
F=base+sf+['breadth','weighted','adx8','disp8','atr8','e20d8','e50d8'];D=pd.DataFrame(rows,columns=['ci','ei','ts','sellret','holdret','delta','j','j2']+F).replace([np.inf,-np.inf],np.nan).fillna(0);print('EVENTS24|%d|trades=%d|features=%d|delta_med=%.4f'%(len(D),D.ci.nunique(),len(F),D.delta.median()),flush=True)
folds=[(430,430,600),(600,600,760),(760,760,920),(920,920,N)];out=[]
for fi,(trainend,vstart,vend) in enumerate(folds,1):
 fit=D.ci<trainend
 for mn in ['ET','HGB']:
  m=ExtraTreesRegressor(n_estimators=400,max_depth=10,min_samples_leaf=10,n_jobs=-1,random_state=240+fi) if mn=='ET' else HistGradientBoostingRegressor(max_iter=250,max_leaf_nodes=15,learning_rate=.04,l2_regularization=3,random_state=240+fi)
  m.fit(D.loc[fit,F],D.loc[fit,'delta'])
  # threshold chosen on latest 25% of prior trades, maximizing realized log return of recursive gate
  c0=int(trainend*.75);thresholds=[0,.0025,.005,.01,.015,.02,.03];best=None
  for th in thresholds:
   rr=[];ign=[]
   for ci in range(c0,trainend):
    q=D[D.ci==ci].sort_values('ei');
    if q.empty:continue
    chosen=None
    for _,r in q.iterrows():
     pred=float(m.predict(pd.DataFrame([r[F].values],columns=F))[0])
     if pred<=th:chosen=r;break
     ign.append(1)
    if chosen is None:chosen=q.iloc[-1]
    rr.append(float(chosen.sellret))
   if not rr:continue
   a=np.clip(rr,-.95,None);sc=np.mean(np.log1p(a));
   if best is None or sc>best[0]:best=(sc,th)
  th=best[1];rb=[];rg=[];ignored=0;events=0
  for ci in range(vstart,vend):
   q=D[D.ci==ci].sort_values('ei');
   if q.empty:continue
   rb.append(float(q.iloc[0].sellret));chosen=None
   for _,r in q.iterrows():
    events+=1;pred=float(m.predict(pd.DataFrame([r[F].values],columns=F))[0])
    if pred<=th:chosen=r;break
    ignored+=1
   if chosen is None:chosen=q.iloc[-1]
   rg.append(float(chosen.sellret))
  if not rg:continue
  b=np.clip(rb,-.95,None);g=np.clip(rg,-.95,None);bl=np.mean(np.log1p(b));gl=np.mean(np.log1p(g));print('FOLD24|%d|%s|thr=%.4f|n=%d|basegeo=%.4f|gategeo=%.4f|delta=%.4f|basemed=%.4f|gatemed=%.4f|basep10=%.4f|gatep10=%.4f|ignored=%.3f'%(fi,mn,th,len(g),np.exp(bl)-1,np.exp(gl)-1,gl-bl,np.median(b),np.median(g),np.quantile(b,.1),np.quantile(g,.1),ignored/max(events,1)),flush=True);out.append((mn,fi,np.exp(bl)-1,np.exp(gl)-1,gl-bl,np.median(b),np.median(g),np.quantile(b,.1),np.quantile(g,.1),ignored/max(events,1)))
A=pd.DataFrame(out,columns=['m','fold','basegeo','gategeo','dlog','basemed','gatemed','basep10','gatep10','ignored'])
for mn,q in A.groupby('m'):print('AGG24|%s|basegeo=%.4f|gategeo=%.4f|dlog=%.5f|basemed=%.4f|gatemed=%.4f|basep10=%.4f|gatep10=%.4f|ignored=%.3f|wins=%d/4'%(mn,q.basegeo.median(),q.gategeo.median(),q.dlog.median(),q.basemed.median(),q.gatemed.median(),q.basep10.median(),q.gatep10.median(),q.ignored.median(),int((q.dlog>0).sum())),flush=True)
print('DONE24',flush=True)