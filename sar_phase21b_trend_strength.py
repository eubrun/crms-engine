"""Phase21B: Phase21 optimal stopping + causal trend-strength features + false-exit classifier."""
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier,HistGradientBoostingRegressor,HistGradientBoostingClassifier
exec(open('sar_phase18c_optimizer.py').read().split("# Candidate rule descriptors")[0])
N=len(O);TF=[1,2,4,6,8,12];print('OOS21B|n=%d'%N,flush=True)

def adx_feats(df,n=14):
 h,l,c=df.high,df.low,df.close;up=h.diff();dn=-l.diff();tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
 atr=tr.ewm(alpha=1/n,adjust=False).mean();pdm=up.where((up>dn)&(up>0),0);mdm=dn.where((dn>up)&(dn>0),0)
 pdi=100*pdm.ewm(alpha=1/n,adjust=False).mean()/atr.replace(0,np.nan);mdi=100*mdm.ewm(alpha=1/n,adjust=False).mean()/atr.replace(0,np.nan);dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan);adx=dx.ewm(alpha=1/n,adjust=False).mean();return adx,pdi,mdi,atr
H={}
for sym in sorted(set(x['sym'] for x in O)):
 xs=[x for x in O if x['sym']==sym];st=min(x['ts'] for x in xs)-pd.Timedelta(days=8);en=max(x['ts'] for x in xs)+pd.Timedelta(hours=1081)
 try:H[sym]=hourly(sym,int(st.timestamp()*1000),int(en.timestamp()*1000));print('H21B|%s|%d'%(sym,len(H[sym])),flush=True)
 except Exception as e:H[sym]=pd.DataFrame();print('FAIL21B|%s|%s'%(sym,str(e)[:100]),flush=True)
rows=[]
for ci,x in enumerate(O):
 p=H.get(x['sym'],pd.DataFrame());p=p[(p.index>=x['ts'])&(p.index<x['ts']+pd.Timedelta(hours=1081))]
 if p.empty:continue
 end=x['sell'] if x['sell'] is not None else p.index[-1];p=p[p.index<=end]
 if len(p)<72:continue
 entry=float(x['entry']);rh=p.high.cummax();r1=p.close.pct_change();top=float(p.high.max());states={}
 for tf in TF:
  z=bars(p,tf);bb,ss=psar_state(z);states[tf]=(z,bb,ss)
 strength={}
 for tf in [4,8,12,24]:
  z=bars(p,tf);adx,pdi,mdi,atr=adx_feats(z);e20=z.close.ewm(span=20,adjust=False).mean();e50=z.close.ewm(span=50,adjust=False).mean();strength[tf]=(z,adx,pdi,mdi,atr,e20,e50)
 prevbread=None
 for j in range(48,len(p),4):
  t=p.index[j];px=float(p.close.iloc[j]);future=p.iloc[j:];fhi=float(future.high.max());miss=max(0,fhi/px-1);ret=px/entry-1;give=max(0,1-px/top);utility=ret-.65*give-max(0,miss-.05)-max(0,miss-.10)-max(0,miss-.15)
  feat=[ret,px/float(rh.iloc[j])-1,float(r1.iloc[max(0,j-23):j+1].std()),float(r1.iloc[max(0,j-47):j+1].std()),float(p.close.iloc[j]/p.close.iloc[j-6]-1),float(p.close.iloc[j]/p.close.iloc[j-12]-1),float(p.close.iloc[j]/p.close.iloc[j-24]-1),float(p.close.iloc[j]/p.close.iloc[max(0,j-72)]-1),j/24]
  bears=[]
  for tf in TF:
   z,bb,ss=states[tf];k=z.index.searchsorted(t,'right')-1
   if k<0:feat += [0,0,999,0];bears.append(0);continue
   bear=float(not bb[k]);dist=(px-float(ss[k]))/px if np.isfinite(ss[k]) else 0;fl=np.where(bb[1:]!=bb[:-1])[0]+1;pr=fl[fl<=k];since=(t-z.index[pr[-1]]).total_seconds()/3600 if len(pr) else 999;feat += [bear,dist,min(since,999),float(since<=12)];bears.append(bear)
  breadth=sum(bears);feat += [breadth,sum((i+1)*v for i,v in enumerate(bears))/21,0 if prevbread is None else breadth-prevbread];prevbread=breadth
  # trend strength multi-TF
  for tf in [4,8,12,24]:
   z,adx,pdi,mdi,atr,e20,e50=strength[tf];k=z.index.searchsorted(t,'right')-1
   if k<2:feat += [0]*9;continue
   A=float(adx.iloc[k]) if np.isfinite(adx.iloc[k]) else 0;Ap=float(adx.iloc[k-1]) if np.isfinite(adx.iloc[k-1]) else A;P=float(pdi.iloc[k]) if np.isfinite(pdi.iloc[k]) else 0;M=float(mdi.iloc[k]) if np.isfinite(mdi.iloc[k]) else 0;AT=float(atr.iloc[k]) if np.isfinite(atr.iloc[k]) else 0;E20=float(e20.iloc[k]);E50=float(e50.iloc[k]);feat += [A,A-Ap,P-M,(px-E20)/px,(px-E50)/px,(E20-float(e20.iloc[k-1]))/px,(E50-float(e50.iloc[k-1]))/px,AT/px,float(z.close.iloc[k]/z.close.iloc[max(0,k-3)]-1)]
  # hourly volume/structure/new-high persistence
  v24=float(p.volume.iloc[max(0,j-23):j+1].mean());v72=float(p.volume.iloc[max(0,j-71):j+1].mean());relv=float(p.volume.iloc[j]/v72) if v72 else 0
  hh24=float(p.high.iloc[j]>=p.high.iloc[max(0,j-23):j+1].max());hh48=float(p.high.iloc[j]>=p.high.iloc[max(0,j-47):j+1].max());sincehi=(t-p.high.iloc[:j+1].idxmax()).total_seconds()/3600
  feat += [v24/v72 if v72 else 0,relv,hh24,hh48,min(sincehi,999)]
  rows.append([ci,t,px,utility,miss,ret,give,int(miss>.10)]+feat)
base=['ret0','dd','vol24','vol48','mom6','mom12','mom24','mom72','age'];sf=[]
for tf in TF:sf += ['bear%d'%tf,'psard%d'%tf,'sinceflip%d'%tf,'recentflip%d'%tf]
F=base+sf+['bearbreadth','bearweighted','breadthdelta']
for tf in [4,8,12,24]:F += ['adx%d'%tf,'adxslope%d'%tf,'disp%d'%tf,'e20dist%d'%tf,'e50dist%d'%tf,'e20slope%d'%tf,'e50slope%d'%tf,'atr%d'%tf,'tfmom%d'%tf]
F += ['volratio24_72','relvol','hh24','hh48','sincehigh']
cols=['ci','ts','px','utility','miss','realret','give','false10']+F;D=pd.DataFrame(rows,columns=cols).replace([np.inf,-np.inf],np.nan).fillna(0);D['rank']=D.groupby('ci').utility.rank(pct=True);print('ROWS21B|%d|features=%d|false10=%.3f'%(len(D),len(F),D.false10.mean()),flush=True)
folds=[(430,430,600),(600,600,760),(760,760,920),(920,920,N)];out=[]
for fi,(trainend,vstart,vend) in enumerate(folds,1):
 cal0=int(trainend*.8);fit=D.ci<cal0
 for mn in ['ET','HGB']:
  if mn=='ET': mr=ExtraTreesRegressor(n_estimators=300,max_depth=13,min_samples_leaf=12,n_jobs=-1,random_state=121+fi);mc=ExtraTreesClassifier(n_estimators=300,max_depth=13,min_samples_leaf=12,class_weight='balanced',n_jobs=-1,random_state=221+fi)
  else: mr=HistGradientBoostingRegressor(max_iter=220,max_leaf_nodes=15,learning_rate=.05,l2_regularization=2,random_state=121+fi);mc=HistGradientBoostingClassifier(max_iter=220,max_leaf_nodes=15,learning_rate=.05,l2_regularization=2,random_state=221+fi)
  mr.fit(D.loc[fit,F],D.loc[fit,'rank']);mc.fit(D.loc[fit,F],D.loc[fit,'false10']);best=None
  for th in [.50,.55,.60,.65,.70,.75]:
   for alpha in [.15,.25,.35,.50,.70]:
    for pers in [1,2,3]:
     rr=[]
     for ci in range(cal0,trainend):
      q=D[D.ci==ci].copy()
      if q.empty:continue
      score=mr.predict(q[F])-alpha*mc.predict_proba(q[F])[:,1];s=q[(pd.Series(score,index=q.index)>=th).rolling(pers).sum()>=pers]
      if s.empty:continue
      r=s.iloc[0];rr.append((r.realret,r['miss'],r.give))
     if not rr:continue
     ar=np.array(rr);cov=len(rr)/(trainend-cal0)
     if cov<.50:continue
     sc=np.median(ar[:,0])-1.4*np.median(ar[:,1])-1.0*np.mean(ar[:,1]>.10)-.6*np.mean(ar[:,1]>.15)+.1*cov
     if best is None or sc>best[0]:best=(sc,th,alpha,pers)
  if best is None:print('FOLD21B|%d|%s|NO_CAL'%(fi,mn),flush=True);continue
  _,th,alpha,pers=best;rr=[]
  for ci in range(vstart,vend):
   q=D[D.ci==ci].copy()
   if q.empty:continue
   score=mr.predict(q[F])-alpha*mc.predict_proba(q[F])[:,1];s=q[(pd.Series(score,index=q.index)>=th).rolling(pers).sum()>=pers]
   if s.empty:continue
   r=s.iloc[0];rr.append((r.realret,r['miss'],r.give))
  if not rr:continue
  ar=np.array(rr);cov=len(rr)/(vend-vstart);print('FOLD21B|%d|%s|thr=%.2f|alpha=%.2f|pers=%d|n=%d|cov=%.3f|ret=%.4f|miss=%.4f|give=%.4f|miss10=%.3f|miss15=%.3f'%(fi,mn,th,alpha,pers,len(rr),cov,np.median(ar[:,0]),np.median(ar[:,1]),np.median(ar[:,2]),np.mean(ar[:,1]>.10),np.mean(ar[:,1]>.15)),flush=True);out.append((mn,fi,cov,np.median(ar[:,0]),np.median(ar[:,1]),np.median(ar[:,2]),np.mean(ar[:,1]>.10),np.mean(ar[:,1]>.15)))
A=pd.DataFrame(out,columns=['model','fold','cov','ret','miss','give','miss10','miss15'])
for mn,q in A.groupby('model'):print('AGG21B|%s|folds=%d|cov=%.3f|ret=%.4f|miss=%.4f|give=%.4f|miss10=%.3f|miss15=%.3f'%(mn,len(q),q['cov'].median(),q['ret'].median(),q['miss'].median(),q['give'].median(),q['miss10'].median(),q['miss15'].median()),flush=True)
print('DONE21B',flush=True)