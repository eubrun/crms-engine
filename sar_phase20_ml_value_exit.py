"""Phase20: ML value-of-HOLD exit. Fixed BUY=Phase17C daily price>PSAR.
Predict future upside and adverse excursion with regressors, then choose causal exit when predicted HOLD value turns negative.
Walk-forward style chronological train/validation/test; final 20% held out within this phase.
"""
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
exec(open('sar_phase18c_optimizer.py').read().split("# Candidate rule descriptors")[0])
n=len(O);a=int(n*.55);b=int(n*.80);print('OOS20|n=%d|train=%d|val=%d|test=%d'%(n,a,b-a,n-b),flush=True)
TF=[1,2,4,6,8,12];H={}
for sym in sorted(set(x['sym'] for x in O)):
 xs=[x for x in O if x['sym']==sym];st=min(x['ts'] for x in xs)-pd.Timedelta(days=5);en=max(x['ts'] for x in xs)+pd.Timedelta(hours=1081)
 try:H[sym]=hourly(sym,int(st.timestamp()*1000),int(en.timestamp()*1000));print('H20|%s|%d'%(sym,len(H[sym])),flush=True)
 except Exception as e:H[sym]=pd.DataFrame();print('FAIL20|%s|%s'%(sym,str(e)[:100]),flush=True)
rows=[]
for ci,x in enumerate(O):
 p=H.get(x['sym'],pd.DataFrame());p=p[(p.index>=x['ts'])&(p.index<x['ts']+pd.Timedelta(hours=1081))]
 if p.empty:continue
 end=x['sell'] if x['sell'] is not None else p.index[-1];p=p[p.index<=end]
 if len(p)<48:continue
 entry=float(x['entry']);rh=p.high.cummax();r1=p.close.pct_change();states={}
 for tf in TF:
  z=bars(p,tf);bb,_=psar_state(z);states[tf]=(z,bb)
 for j in range(24,len(p)-1,4):
  t=p.index[j];px=float(p.close.iloc[j]);future=p.iloc[j+1:];fhi=float(future.high.max());flo=float(future.low.min());up=max(0.,fhi/px-1);down=max(0.,1-flo/px)
  feat=[px/entry-1,px/float(rh.iloc[j])-1,float(r1.iloc[max(0,j-23):j+1].std()),float(p.close.iloc[j]/p.close.iloc[j-6]-1),float(p.close.iloc[j]/p.close.iloc[j-24]-1),float(p.close.iloc[j]/p.close.iloc[max(0,j-72)]-1),j/24]
  for tf in TF:
   z,bb=states[tf];k=z.index.searchsorted(t,'right')-1;feat.append(0 if k<0 else float(not bb[k]))
  rows.append([ci,t,px,up,down]+feat)
cols=['ci','ts','px','up','down','ret','dd','vol24','mom6','mom24','mom72','age']+['bear%d'%t for t in TF]
D=pd.DataFrame(rows,columns=cols);F=cols[5:];X=D[F].replace([np.inf,-np.inf],np.nan).fillna(0);print('ROWS20|%d'%len(D),flush=True)
tr=D.ci<a;va=(D.ci>=a)&(D.ci<b);te=D.ci>=b
models=[('HGB',HistGradientBoostingRegressor(max_iter=250,max_leaf_nodes=15,learning_rate=.05,l2_regularization=2,loss='absolute_error',random_state=20)),('ET',ExtraTreesRegressor(n_estimators=350,max_depth=12,min_samples_leaf=15,random_state=20,n_jobs=-1))]
for mn,base in models:
 import copy;mu=copy.deepcopy(base);md=copy.deepcopy(base);mu.fit(X[tr],D.loc[tr,'up']);md.fit(X[tr],D.loc[tr,'down'])
 # validation searches economic risk aversion lambda and margin. Exit when predicted upside - lambda*predicted downside < margin.
 best=None
 for lam in [.5,.75,1,1.25,1.5,2,2.5,3]:
  for margin in [-.03,-.02,-.01,0,.01,.02,.03]:
   rr=[]
   for ci in range(a,b):
    q=D[D.ci==ci];
    if q.empty:continue
    xx=q[F].replace([np.inf,-np.inf],np.nan).fillna(0);v=mu.predict(xx)-lam*md.predict(xx);s=q[v<margin]
    if s.empty:continue
    r=s.iloc[0];x=O[ci];pp=H[x['sym']];pp=pp[(pp.index>=x['ts'])&(pp.index<=x['sell'])] if x['sell'] is not None else pp[pp.index>=x['ts']];top=float(pp.high.max());rr.append((r.px/float(x['entry'])-1,max(0,top/r.px-1)))
   if not rr:continue
   ar=np.array(rr);cov=len(rr)/(b-a);score=np.median(ar[:,0])-1.5*np.median(ar[:,1])-.75*np.mean(ar[:,1]>.10)+.15*cov
   if best is None or score>best[0]:best=(score,lam,margin,cov)
 sc,lam,margin,cv=best;res=[]
 for ci in range(b,n):
  q=D[D.ci==ci];
  if q.empty:continue
  xx=q[F].replace([np.inf,-np.inf],np.nan).fillna(0);v=mu.predict(xx)-lam*md.predict(xx);s=q[v<margin]
  if s.empty:continue
  r=s.iloc[0];x=O[ci];pp=H[x['sym']];pp=pp[(pp.index>=x['ts'])&(pp.index<=x['sell'])] if x['sell'] is not None else pp[pp.index>=x['ts']];top=float(pp.high.max());res.append((r.px/float(x['entry'])-1,max(0,top/r.px-1),(x['sell']-r.ts).total_seconds()/3600 if x['sell'] is not None else np.nan))
 ar=np.array(res);print('TEST20|%s|lambda=%.2f|margin=%.3f|n=%d|cov=%.3f|ret_med=%.4f|miss_med=%.4f|miss5=%.3f|miss10=%.3f|miss15=%.3f|daily_lead=%.1f'%(mn,lam,margin,len(res),len(res)/(n-b),np.median(ar[:,0]),np.median(ar[:,1]),np.mean(ar[:,1]>.05),np.mean(ar[:,1]>.10),np.mean(ar[:,1]>.15),np.nanmedian(ar[:,2])),flush=True)
 if mn=='ET':
  imp=sorted(zip(F,(mu.feature_importances_+md.feature_importances_)/2),key=lambda z:z[1],reverse=True);print('IMPORT20|'+'|'.join('%s=%.3f'%z for z in imp),flush=True)
print('DONE20',flush=True)