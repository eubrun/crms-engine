"""Phase21: walk-forward ML optimal stopping surrogate. Fixed daily PSAR entry."""
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
exec(open('sar_phase18c_optimizer.py').read().split("# Candidate rule descriptors")[0])
N=len(O);TF=[1,2,4,6,8,12];print('OOS21|n=%d'%N,flush=True)
H={}
for sym in sorted(set(x['sym'] for x in O)):
 xs=[x for x in O if x['sym']==sym];st=min(x['ts'] for x in xs)-pd.Timedelta(days=5);en=max(x['ts'] for x in xs)+pd.Timedelta(hours=1081)
 try:H[sym]=hourly(sym,int(st.timestamp()*1000),int(en.timestamp()*1000));print('H21|%s|%d'%(sym,len(H[sym])),flush=True)
 except Exception as e:H[sym]=pd.DataFrame();print('FAIL21|%s|%s'%(sym,str(e)[:100]),flush=True)
rows=[]
for ci,x in enumerate(O):
 p=H.get(x['sym'],pd.DataFrame());p=p[(p.index>=x['ts'])&(p.index<x['ts']+pd.Timedelta(hours=1081))]
 if p.empty:continue
 end=x['sell'] if x['sell'] is not None else p.index[-1];p=p[p.index<=end]
 if len(p)<48:continue
 entry=float(x['entry']);rh=p.high.cummax();r1=p.close.pct_change();top=float(p.high.max());states={}
 for tf in TF:
  z=bars(p,tf);bb,ss=psar_state(z);states[tf]=(z,bb,ss)
 for j in range(24,len(p),4):
  t=p.index[j];px=float(p.close.iloc[j]);future=p.iloc[j:];fhi=float(future.high.max());miss=max(0,fhi/px-1);ret=px/entry-1;give=max(0,1-px/top)
  # utility deliberately tolerates first 5% future upside; heavily penalizes >10/15% premature exits
  utility=ret-0.65*give-1.0*max(0,miss-.05)-1.0*max(0,miss-.10)-1.0*max(0,miss-.15)
  feat=[ret,px/float(rh.iloc[j])-1,float(r1.iloc[max(0,j-23):j+1].std()),float(r1.iloc[max(0,j-47):j+1].std()),float(p.close.iloc[j]/p.close.iloc[j-6]-1),float(p.close.iloc[j]/p.close.iloc[j-12]-1),float(p.close.iloc[j]/p.close.iloc[j-24]-1),float(p.close.iloc[j]/p.close.iloc[max(0,j-72)]-1),j/24]
  bears=[]
  for tf in TF:
   z,bb,ss=states[tf];k=z.index.searchsorted(t,'right')-1
   if k<0: feat += [0,0,999,0];bears.append(0);continue
   bear=float(not bb[k]);dist=(px-float(ss[k]))/px if np.isfinite(ss[k]) else 0
   flips=np.where(bb[1:]!=bb[:-1])[0]+1;prev=flips[flips<=k];since=(t-z.index[prev[-1]]).total_seconds()/3600 if len(prev) else 999
   recent=float(len(prev)>0 and since<=12);feat += [bear,dist,min(since,999),recent];bears.append(bear)
  feat += [sum(bears),sum((i+1)*v for i,v in enumerate(bears))/21]
  rows.append([ci,t,px,utility,miss,ret,give]+feat)
base=['ret0','dd','vol24','vol48','mom6','mom12','mom24','mom72','age'];ff=[]
for tf in TF:ff += ['bear%d'%tf,'psard%d'%tf,'sinceflip%d'%tf,'recentflip%d'%tf]
F=base+ff+['bearbreadth','bearweighted'];cols=['ci','ts','px','utility','miss','realret','give']+F
D=pd.DataFrame(rows,columns=cols).replace([np.inf,-np.inf],np.nan).fillna(0);print('ROWS21|%d|features=%d'%(len(D),len(F)),flush=True)
# normalize target to within-trade percentile rank: learning relative exit quality
D['rank']=D.groupby('ci').utility.rank(pct=True)
# expanding walk-forward: train end / validation block. Threshold selected inside trailing 20% of training, evaluated next block.
folds=[(430,430,600),(600,600,760),(760,760,920),(920,920,N)]
allres=[]
for fi,(trainend,vstart,vend) in enumerate(folds,1):
 tr=D.ci<trainend;ev=(D.ci>=vstart)&(D.ci<vend)
 # inner calibration = last 20% of chronological train; model fit before calibration
 cal0=int(trainend*.8);fit=D.ci<cal0;cal=(D.ci>=cal0)&(D.ci<trainend)
 for mn,m in [('ET',ExtraTreesRegressor(n_estimators=300,max_depth=12,min_samples_leaf=12,n_jobs=-1,random_state=21+fi)),('HGB',HistGradientBoostingRegressor(max_iter=220,max_leaf_nodes=15,learning_rate=.05,l2_regularization=2,random_state=21+fi))]:
  m.fit(D.loc[fit,F],D.loc[fit,'rank'])
  # choose score threshold and persistence on calibration; enforce >=50% coverage to prevent degenerate solutions
  best=None
  for th in [.55,.60,.65,.70,.75,.80,.85]:
   for pers in [1,2,3]:
    rr=[]
    for ci in range(cal0,trainend):
     q=D[D.ci==ci].copy()
     if q.empty:continue
     q['s']=m.predict(q[F]);hit=(q.s>=th).rolling(pers).sum()>=pers;s=q[hit]
     if s.empty:continue
     r=s.iloc[0];rr.append((r.realret,r.miss,r.give))
    if not rr:continue
    ar=np.array(rr);cov=len(rr)/(trainend-cal0)
    if cov<.50:continue
    sc=np.median(ar[:,0])-1.25*np.median(ar[:,1])-.6*np.mean(ar[:,1]>.10)-.3*np.mean(ar[:,1]>.15)+.1*cov
    if best is None or sc>best[0]:best=(sc,th,pers,cov)
  if best is None:print('FOLD21|%d|%s|NO_CAL',fi,mn,flush=True);continue
  _,th,pers,_=best;rr=[]
  for ci in range(vstart,vend):
   q=D[D.ci==ci].copy()
   if q.empty:continue
   q['s']=m.predict(q[F]);hit=(q.s>=th).rolling(pers).sum()>=pers;s=q[hit]
   if s.empty:continue
   r=s.iloc[0];x=O[ci];lead=(x['sell']-r.ts).total_seconds()/3600 if x['sell'] is not None else np.nan;rr.append((r.realret,r.miss,r.give,lead))
  ar=np.array(rr);cov=len(rr)/(vend-vstart)
  print('FOLD21|%d|%s|thr=%.2f|pers=%d|n=%d|cov=%.3f|ret=%.4f|miss=%.4f|give=%.4f|miss5=%.3f|miss10=%.3f|miss15=%.3f|lead=%.1f'%(fi,mn,th,pers,len(rr),cov,np.median(ar[:,0]),np.median(ar[:,1]),np.median(ar[:,2]),np.mean(ar[:,1]>.05),np.mean(ar[:,1]>.10),np.mean(ar[:,1]>.15),np.nanmedian(ar[:,3])),flush=True)
  allres.append((mn,fi,cov,np.median(ar[:,0]),np.median(ar[:,1]),np.median(ar[:,2]),np.mean(ar[:,1]>.10),np.mean(ar[:,1]>.15)))
  if mn=='ET':
   imp=sorted(zip(F,m.feature_importances_),key=lambda z:z[1],reverse=True)[:12];print('IMPORT21|fold=%d|'%fi+'|'.join('%s=%.3f'%z for z in imp),flush=True)
A=pd.DataFrame(allres,columns=['model','fold','cov','ret','miss','give','miss10','miss15'])
for mn,q in A.groupby('model'):
 print('AGG21|%s|folds=%d|cov_med=%.3f|ret_med=%.4f|miss_med=%.4f|give_med=%.4f|miss10_med=%.3f|miss15_med=%.3f'%(mn,len(q),q.cov.median(),q.ret.median(),q['miss'].median(),q.give.median(),q.miss10.median(),q.miss15.median()),flush=True)
print('DONE21',flush=True)