"""Phase22: causal Gann timing/geometry ablation on Phase21 optimal-stopping framework.
ENTRY fixed: Phase17C daily price>PSAR. Compare baseline vs +Gann per asset.
"""
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
exec(open('sar_phase18c_optimizer.py').read().split("# Candidate rule descriptors")[0])
N=len(O);TF=[1,2,4,6,8,12];CYC=np.array([7,14,21,30,45,60,90],float);print('OOS22|n=%d'%N,flush=True)
H={}
for sym in sorted(set(x['sym'] for x in O)):
 xs=[x for x in O if x['sym']==sym];st=min(x['ts'] for x in xs)-pd.Timedelta(days=120);en=max(x['ts'] for x in xs)+pd.Timedelta(hours=1081)
 try:H[sym]=hourly(sym,int(st.timestamp()*1000),int(en.timestamp()*1000));print('H22|%s|%d'%(sym,len(H[sym])),flush=True)
 except Exception as e:H[sym]=pd.DataFrame();print('FAIL22|%s|%s'%(sym,str(e)[:100]),flush=True)

def atr(x,n=24):
 tr=pd.concat([x.high-x.low,(x.high-x.close.shift()).abs(),(x.low-x.close.shift()).abs()],axis=1).max(axis=1);return tr.rolling(n,min_periods=5).mean()
def pivot_feats(hist,t,px):
 # causal confirmed daily pivots: candidate must be >=3 completed daily bars old; confirmation uses only bars already known at t
 d=hist.resample('24h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna();d=d[d.index<t.floor('D')]
 if len(d)<10:return [999,999,1,1,0,0,0,0]
 hi=d.high;lo=d.low;ph=[];pl=[]
 for k in range(3,len(d)-3):
  if hi.iloc[k]>=hi.iloc[k-3:k+4].max():ph.append(k)
  if lo.iloc[k]<=lo.iloc[k-3:k+4].min():pl.append(k)
 kh=ph[-1] if ph else max(0,len(d)-8);kl=pl[-1] if pl else max(0,len(d)-8);th=d.index[kh];tl=d.index[kl];dh=max(0,(t-th).total_seconds()/86400);dl=max(0,(t-tl).total_seconds()/86400)
 def cyc(days):
  dist=np.abs(CYC-days);i=int(dist.argmin());return dist[i]/CYC[i],CYC[i]
 ch,idh=cyc(dh);cl,idl=cyc(dl);A=float(atr(hist).iloc[-1]) if len(hist) else 0;A=A if np.isfinite(A) and A>0 else max(px*.02,1e-9)
 # ATR-normalized 1x1-style projections: 1 ATR/day from pivot; distance to projection
 projh=float(hi.iloc[kh])-A*dh;projl=float(lo.iloc[kl])+A*dl;gh=(px-projh)/A;gl=(px-projl)/A
 # square-root price phase proximity, scale-free fractional distance to nearest integer/half-integer phase
 root=np.sqrt(max(px,1e-12));sq=min(abs(root-round(root)),abs(root*2-round(root*2))/2)
 conf=min(ch,cl)+min(abs(gh),abs(gl))/10
 return [dh,dl,ch,cl,idh/90,idl/90,np.tanh(gh/10),np.tanh(gl/10),sq,conf]
rows=[]
for ci,x in enumerate(O):
 full=H.get(x['sym'],pd.DataFrame());p=full[(full.index>=x['ts'])&(full.index<x['ts']+pd.Timedelta(hours=1081))]
 if p.empty:continue
 end=x['sell'] if x['sell'] is not None else p.index[-1];p=p[p.index<=end]
 if len(p)<48:continue
 entry=float(x['entry']);rh=p.high.cummax();r1=p.close.pct_change();top=float(p.high.max());states={}
 for tf in TF:
  z=bars(p,tf);bb,ss=psar_state(z);states[tf]=(z,bb,ss)
 for j in range(24,len(p),4):
  t=p.index[j];px=float(p.close.iloc[j]);future=p.iloc[j:];fhi=float(future.high.max());miss=max(0,fhi/px-1);ret=px/entry-1;give=max(0,1-px/top);u=ret-.65*give-max(0,miss-.05)-max(0,miss-.10)-max(0,miss-.15)
  b=[ret,px/float(rh.iloc[j])-1,float(r1.iloc[max(0,j-23):j+1].std()),float(r1.iloc[max(0,j-47):j+1].std()),float(p.close.iloc[j]/p.close.iloc[j-6]-1),float(p.close.iloc[j]/p.close.iloc[j-12]-1),float(p.close.iloc[j]/p.close.iloc[j-24]-1),float(p.close.iloc[j]/p.close.iloc[max(0,j-72)]-1),j/24];bears=[]
  for tf in TF:
   z,bb,ss=states[tf];k=z.index.searchsorted(t,'right')-1
   if k<0:b += [0,0,999,0];bears.append(0);continue
   be=float(not bb[k]);dist=(px-float(ss[k]))/px if np.isfinite(ss[k]) else 0;fl=np.where(bb[1:]!=bb[:-1])[0]+1;pr=fl[fl<=k];since=(t-z.index[pr[-1]]).total_seconds()/3600 if len(pr) else 999;b += [be,dist,min(since,999),float(since<=12)];bears.append(be)
  b += [sum(bears),sum((i+1)*v for i,v in enumerate(bears))/21]
  hist=full[full.index<=t];g=pivot_feats(hist,t,px);age=(t-x['ts']).total_seconds()/86400;cd=np.abs(CYC-age);ii=int(cd.argmin());g += [cd[ii]/CYC[ii],CYC[ii]/90]
  rows.append([ci,t,u,miss,ret,give]+b+g)
base=['ret0','dd','vol24','vol48','mom6','mom12','mom24','mom72','age'];sf=[]
for tf in TF:sf += ['bear%d'%tf,'psard%d'%tf,'sinceflip%d'%tf,'recentflip%d'%tf]
B=base+sf+['bearbreadth','bearweighted'];G=['daysSwingHigh','daysSwingLow','cycleDistHigh','cycleDistLow','cycleHighId','cycleLowId','gannGeomHigh','gannGeomLow','sqrtPhase','gannConfluence','buyCycleDist','buyCycleId'];F=B+G
D=pd.DataFrame(rows,columns=['ci','ts','utility','miss','realret','give']+F).replace([np.inf,-np.inf],np.nan).fillna(0);D['rank']=D.groupby('ci').utility.rank(pct=True);print('ROWS22|%d|base=%d|gann=%d'%(len(D),len(B),len(G)),flush=True)
folds=[(430,430,600),(600,600,760),(760,760,920),(920,920,N)];out=[]
for variant,features in [('BASE',B),('GANN',F)]:
 for fi,(trainend,vstart,vend) in enumerate(folds,1):
  cal0=int(trainend*.8);fit=D.ci<cal0;m=HistGradientBoostingRegressor(max_iter=220,max_leaf_nodes=15,learning_rate=.05,l2_regularization=2,random_state=220+fi);m.fit(D.loc[fit,features],D.loc[fit,'rank']);best=None
  for th in [.55,.60,.65,.70,.75,.80,.85]:
   for pers in [1,2,3]:
    rr=[]
    for ci in range(cal0,trainend):
     q=D[D.ci==ci].copy();
     if q.empty:continue
     s=q[(pd.Series(m.predict(q[features]),index=q.index)>=th).rolling(pers).sum()>=pers]
     if s.empty:continue
     r=s.iloc[0];rr.append((r.realret,r['miss'],r.give))
    if not rr:continue
    a=np.array(rr);cov=len(rr)/(trainend-cal0)
    if cov<.50:continue
    sc=np.median(a[:,0])-1.25*np.median(a[:,1])-.6*np.mean(a[:,1]>.10)-.3*np.mean(a[:,1]>.15)+.1*cov
    if best is None or sc>best[0]:best=(sc,th,pers)
  if best is None:print('FOLD22|%s|%d|NO_CAL'%(variant,fi),flush=True);continue
  _,th,pers=best;rr=[]
  for ci in range(vstart,vend):
   q=D[D.ci==ci].copy();
   if q.empty:continue
   s=q[(pd.Series(m.predict(q[features]),index=q.index)>=th).rolling(pers).sum()>=pers]
   if s.empty:continue
   r=s.iloc[0];rr.append((r.realret,r['miss'],r.give))
  if not rr:continue
  a=np.array(rr);cov=len(rr)/(vend-vstart);print('FOLD22|%s|%d|thr=%.2f|pers=%d|n=%d|cov=%.3f|ret=%.4f|miss=%.4f|give=%.4f|miss10=%.3f|miss15=%.3f'%(variant,fi,th,pers,len(rr),cov,np.median(a[:,0]),np.median(a[:,1]),np.median(a[:,2]),np.mean(a[:,1]>.10),np.mean(a[:,1]>.15)),flush=True);out.append((variant,fi,cov,np.median(a[:,0]),np.median(a[:,1]),np.median(a[:,2]),np.mean(a[:,1]>.10),np.mean(a[:,1]>.15)))
A=pd.DataFrame(out,columns=['v','fold','cov','ret','miss','give','m10','m15'])
for v,q in A.groupby('v'):print('AGG22|%s|cov=%.3f|ret=%.4f|miss=%.4f|give=%.4f|miss10=%.3f|miss15=%.3f'%(v,q['cov'].median(),q['ret'].median(),q['miss'].median(),q['give'].median(),q['m10'].median(),q['m15'].median()),flush=True)
print('DONE22',flush=True)