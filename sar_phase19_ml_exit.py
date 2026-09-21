"""Phase 19 — ML exit model.
Fixed entry: Phase17C daily price>PSAR event. Learn exit probability from causal intraday state.
Walk-forward / chronological validation only; future TOP used solely to construct training labels and evaluate exits.
Models: HistGradientBoosting + RandomForest/ExtraTrees when sklearn available. Features include multi-TF PSAR,
returns, volatility, running MFE/drawdown, momentum and time-in-trade. Decision threshold is selected on train/validation,
then frozen for untouched test. Objective: maximize captured cycle return while penalizing giveback and premature exits.
"""
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from sklearn.metrics import roc_auc_score,precision_recall_curve
exec(open('sar_phase18c_optimizer.py').read().split("# Candidate rule descriptors")[0])
# exact Phase17C OOS is already O; chronological 55/20/25 train/validation/test
n=len(O);a=int(n*.55);b=int(n*.75);print('OOS19|n=%d|train=%d|val=%d|test=%d'%(n,a,b-a,n-b),flush=True)
TF=[1,2,4,6,8,12]
H={}
for sym in sorted(set(x['sym'] for x in O)):
 xs=[x for x in O if x['sym']==sym];st=min(x['ts'] for x in xs)-pd.Timedelta(days=5);en=max(x['ts'] for x in xs)+pd.Timedelta(hours=1081)
 try:H[sym]=hourly(sym,int(st.timestamp()*1000),int(en.timestamp()*1000));print('H19|%s|%d'%(sym,len(H[sym])),flush=True)
 except Exception as e:H[sym]=pd.DataFrame();print('FAIL19|%s|%s'%(sym,str(e)[:100]),flush=True)
rows=[];meta=[]
for ci,x in enumerate(O):
 p=H.get(x['sym'],pd.DataFrame());p=p[(p.index>=x['ts'])&(p.index<x['ts']+pd.Timedelta(hours=1081))]
 if p.empty:continue
 end=x['sell'] if x['sell'] is not None else p.index[-1];p=p[p.index<=end]
 if len(p)<24:continue
 entry=float(x['entry']);topi=int(np.argmax(p.high.to_numpy(float)));topt=p.index[topi];top=float(p.high.iloc[topi])
 states={}
 for tf in TF:
  z=bars(p,tf);bb,_=psar_state(z);states[tf]=(z,bb)
 rh=p.high.cummax();ret=p.close/entry-1;dd=p.close/rh-1
 r1=p.close.pct_change();vol24=r1.rolling(24).std();mom6=p.close.pct_change(6);mom24=p.close.pct_change(24);mom72=p.close.pct_change(72)
 # 4-hour sampling limits dependence and size. Label=1 if TOP already occurred OR future upside <=5% and downside/giveback has begun.
 for j in range(24,len(p),4):
  t=p.index[j];px=float(p.close.iloc[j]);future=p.iloc[j:];fhi=float(future.high.max());fup=fhi/px-1
  feat=[float(ret.iloc[j]),float(dd.iloc[j]),float(vol24.iloc[j] or 0),float(mom6.iloc[j]),float(mom24.iloc[j]),float(mom72.iloc[j]),j/24]
  for tf in TF:
   z,bb=states[tf];k=z.index.searchsorted(t,'right')-1;feat.append(0 if k<0 else float(not bb[k]))
  # training target: economically safe exit zone; tolerate <=5% missed upside
  y=int((t>=topt) or (fup<=.05 and dd.iloc[j]<=-.02))
  rows.append([ci,t,px,y]+feat);meta.append((top,entry,end))
cols=['ci','ts','px','y','ret','dd','vol24','mom6','mom24','mom72','age']+['bear%d'%tf for tf in TF]
D=pd.DataFrame(rows,columns=cols);print('ROWS19|%d|pos=%.3f'%(len(D),D.y.mean()),flush=True)
F=cols[4:];X=D[F].replace([np.inf,-np.inf],np.nan).fillna(0);y=D.y
tr=D.ci<a;va=(D.ci>=a)&(D.ci<b);te=D.ci>=b
models=[('HGB',HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=15,learning_rate=.05,l2_regularization=2,random_state=19)),('ET',ExtraTreesClassifier(n_estimators=350,max_depth=10,min_samples_leaf=20,class_weight='balanced',random_state=19,n_jobs=-1))]
for mn,m in models:
 m.fit(X[tr],y[tr]);pv=m.predict_proba(X[va])[:,1];pt=m.predict_proba(X[te])[:,1]
 print('MODEL19|%s|auc_val=%.4f|auc_test=%.4f'%(mn,roc_auc_score(y[va],pv),roc_auc_score(y[te],pt)),flush=True)
 # threshold chosen ONLY on validation by simulated first signal per cycle; score exit return - giveback - premature missed upside.
 best=None
 for th in np.arange(.35,.86,.025):
  vals=[]
  for ci in range(a,b):
   q=D[(D.ci==ci)&va].copy()
   if q.empty:continue
   q['pr']=m.predict_proba(q[F].replace([np.inf,-np.inf],np.nan).fillna(0))[:,1];s=q[q.pr>=th]
   if s.empty:continue
   r=s.iloc[0];x=O[ci];p0=H[x['sym']];pp=p0[(p0.index>=x['ts'])&(p0.index<=x['sell'])] if x['sell'] is not None else p0[p0.index>=x['ts']];top=float(pp.high.max());retx=r.px/float(x['entry'])-1;miss=max(0,top/r.px-1);vals.append((retx,miss))
  if not vals:continue
  ar=np.array(vals);cov=len(vals)/(b-a);score=np.median(ar[:,0])-1.25*np.median(ar[:,1])-.5*np.mean(ar[:,1]>.10)+.15*cov
  if best is None or score>best[0]:best=(score,th,cov)
 th=best[1];res=[]
 for ci in range(b,n):
  q=D[D.ci==ci].copy()
  if q.empty:continue
  q['pr']=m.predict_proba(q[F].replace([np.inf,-np.inf],np.nan).fillna(0))[:,1];s=q[q.pr>=th]
  if s.empty:continue
  r=s.iloc[0];x=O[ci];p0=H[x['sym']];pp=p0[(p0.index>=x['ts'])&(p0.index<=x['sell'])] if x['sell'] is not None else p0[p0.index>=x['ts']];top=float(pp.high.max());retx=r.px/float(x['entry'])-1;miss=max(0,top/r.px-1);res.append((retx,miss,(x['sell']-r.ts).total_seconds()/3600 if x['sell'] is not None else np.nan))
 ar=np.array(res);print('TEST19|%s|thr=%.3f|n=%d|cov=%.3f|ret_med=%.4f|miss_med=%.4f|miss5=%.3f|miss10=%.3f|miss15=%.3f|daily_lead=%.1f'%(mn,th,len(res),len(res)/(n-b),np.median(ar[:,0]),np.median(ar[:,1]),np.mean(ar[:,1]>.05),np.mean(ar[:,1]>.10),np.mean(ar[:,1]>.15),np.nanmedian(ar[:,2])),flush=True)
 if mn=='ET':
  imp=sorted(zip(F,m.feature_importances_),key=lambda z:z[1],reverse=True);print('IMPORT19|'+'|'.join('%s=%.3f'%z for z in imp),flush=True)
print('DONE19',flush=True)