"""Phase30: optimize entry quality while freezing exit rule from Phase29. Entry remains Daily PSAR bullish cross; test causal filters/rankings known at entry. Walk-forward only."""
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
exec(open('sar_phase18c_optimizer.py').read().split("# Candidate rule descriptors")[0])
N=len(O);print('OOS30|n=%d'%N,flush=True)
H={}
for sym in sorted(set(x['sym'] for x in O)):
 xs=[x for x in O if x['sym']==sym];st=min(x['ts'] for x in xs)-pd.Timedelta(days=220);en=max(x['ts'] for x in xs)+pd.Timedelta(hours=1081)
 try:H[sym]=hourly(sym,int(st.timestamp()*1000),int(en.timestamp()*1000));print('H30|%s|%d'%(sym,len(H[sym])),flush=True)
 except Exception as e:H[sym]=pd.DataFrame();print('FAIL30|%s|%s'%(sym,str(e)[:100]),flush=True)
def firstbear(p,tf):
 z=bars(p,tf);b,s=psar_state(z)
 for j in range(6,len(p)):
  k=z.index.searchsorted(p.index[j],'right')-1
  if k>=0 and not b[k]:return j
 return len(p)-1
def atr(df,n=14):
 tr=pd.concat([df.high-df.low,(df.high-df.close.shift()).abs(),(df.low-df.close.shift()).abs()],axis=1).max(axis=1);return tr.ewm(alpha=1/n,adjust=False).mean()
def adx(df,n=14):
 h,l,c=df.high,df.low,df.close;up=h.diff();dn=-l.diff();tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1);a=tr.ewm(alpha=1/n,adjust=False).mean();pp=up.where((up>dn)&(up>0),0).ewm(alpha=1/n,adjust=False).mean();mm=dn.where((dn>up)&(dn>0),0).ewm(alpha=1/n,adjust=False).mean();P=100*pp/a.replace(0,np.nan);M=100*mm/a.replace(0,np.nan);X=100*(P-M).abs()/(P+M).replace(0,np.nan);return X.ewm(alpha=1/n,adjust=False).mean(),P,M
R=[]
for ci,x in enumerate(O):
 full=H.get(x['sym'],pd.DataFrame());hist=full[full.index<=x['ts']]
 if len(hist)<300:continue
 entry=float(x['entry']);zw=bars(hist,168);bw,sw=psar_state(zw);kw=len(zw)-1
 if kw<0:continue
 wb=int(bool(bw[kw]));wd=(entry-float(sw[kw]))/entry if np.isfinite(sw[kw]) else 0
 p=full[(full.index>=x['ts'])&(full.index<x['ts']+pd.Timedelta(hours=1081))]
 if len(p)<48:continue
 j8=firstbear(p,8);j12=firstbear(p,12);j24=firstbear(p,24);r8=float(p.close.iloc[j8]/entry-1);r12=float(p.close.iloc[j12]/entry-1);r24=float(p.close.iloc[j24]/entry-1)
 # frozen Phase29 4%/10% hierarchical exit
 ret=r8 if wb==0 or r8<.04 else (r12 if r8<.10 else r24)
 # causal features at entry
 qd=bars(hist,24);bd,sd=psar_state(qd);X,P,M=adx(qd);A=atr(qd);c=qd.close;e20=c.ewm(span=20,adjust=False).mean();e50=c.ewm(span=50,adjust=False).mean();e100=c.ewm(span=100,adjust=False).mean();k=len(qd)-1
 vv=lambda s: float(s.iloc[k]) if k>=0 and np.isfinite(s.iloc[k]) else 0
 feats=[wb,wd,vv(X),vv(P)-vv(M),vv(A)/entry,(entry-vv(e20))/entry,(entry-vv(e50))/entry,(entry-vv(e100))/entry]
 for lag in [1,3,7,14,30,60,90]: feats.append(float(entry/c.iloc[max(0,k-lag)]-1))
 rr=c.pct_change();feats += [float(rr.tail(7).std()),float(rr.tail(30).std()),float(rr.tail(90).std())]
 # multi-timeframe SAR state/breadth at entry
 breadth=0
 for tf in [4,6,8,12,24,168]:
  z=bars(hist,tf);b,s=psar_state(z);kk=len(z)-1;bull=float(kk>=0 and b[kk]);breadth+=bull;feats += [bull,(entry-float(s[kk]))/entry if kk>=0 and np.isfinite(s[kk]) else 0]
 feats += [breadth]
 R.append([ci,x['sym'],x['ts'],ret,r8,r12,r24]+feats)
fn=['wb','wdist','adx','didiff','atr','e20d','e50d','e100d']+['mom%d'%x for x in [1,3,7,14,30,60,90]]+['vol7','vol30','vol90']
for tf in [4,6,8,12,24,168]:fn += ['bull%d'%tf,'psard%d'%tf]
fn += ['breadth']
D=pd.DataFrame(R,columns=['ci','sym','ts','ret','r8','r12','r24']+fn).replace([np.inf,-np.inf],np.nan).fillna(0);print('ROWS30|%d|features=%d'%(len(D),len(fn)),flush=True)
# univariate filters: report top/bottom halves/quartiles on OOS for interpretability
for f in ['wb','wdist','adx','didiff','atr','e20d','e50d','mom7','mom30','mom90','vol30','breadth']:
 q=D[D.ci>=430].copy()
 if f=='wb': groups=q.groupby(f)
 else:
  q['g']=pd.qcut(q[f],4,labels=False,duplicates='drop');groups=q.groupby('g')
 vals=[]
 for z,g in groups:
  r=np.clip(g.ret.values,-.95,None);vals.append('%s:n%d:g%.3f:m%.3f'%(str(z),len(r),np.exp(np.mean(np.log1p(r)))-1,np.median(r)))
 print('UNI30|%s|%s'%(f,'|'.join(vals)),flush=True)
# Walk-forward ML ranking: trade only top fractions predicted; compare per-trade edge AND portfolio opportunity-adjusted return proxy (nontraded=0)
folds=[(430,600),(600,760),(760,920),(920,N)]
for mn in ['ET','HGB']:
 for frac in [.25,.40,.50,.60,.75]:
  out=[]
  for fi,(te,a,b) in enumerate(folds,1):
   tr=D.ci<te;q=D[(D.ci>=a)&(D.ci<b)]
   if len(q)==0:continue
   y=np.log1p(np.clip(D.loc[tr,'ret'].values,-.95,None))
   m=ExtraTreesRegressor(n_estimators=600,max_depth=8,min_samples_leaf=15,n_jobs=-1,random_state=300+fi) if mn=='ET' else HistGradientBoostingRegressor(max_iter=300,max_leaf_nodes=12,learning_rate=.035,l2_regularization=5,random_state=300+fi)
   m.fit(D.loc[tr,fn],y);ptr=m.predict(D.loc[tr,fn]);pq=m.predict(q[fn]);thr=np.quantile(ptr,1-frac);take=pq>=thr;r=np.clip(q.ret.values,-.95,None);sel=r[take];base=np.mean(np.log1p(r));edge=np.mean(np.log1p(sel)) if len(sel) else -99;opp=np.mean(np.where(take,np.log1p(r),0));out.append((edge-base,opp,base,take.mean(),len(sel)));print('FOLD30|%s|frac=%.2f|f=%d|n=%d|take=%d|rate=%.3f|basegeo=%.4f|selgeo=%.4f|opp=%.5f'%(mn,frac,fi,len(q),len(sel),take.mean(),np.exp(base)-1,np.exp(edge)-1,opp),flush=True)
  A=np.array(out);print('AGG30|%s|frac=%.2f|winsEdge=%d/4|dlogMed=%.5f|oppMed=%.5f|baseLogMed=%.5f|takeMed=%.3f'%(mn,frac,int((A[:,0]>0).sum()),np.median(A[:,0]),np.median(A[:,1]),np.median(A[:,2]),np.median(A[:,3])),flush=True)
print('DONE30',flush=True)