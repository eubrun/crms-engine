"""Phase23: direct realized-PnL tournament for exits after fixed Daily PSAR BUY.
Simple causal policies compete first. Objective = walk-forward log growth, with oracle capture diagnostics.
"""
import numpy as np,pandas as pd
exec(open('sar_phase18c_optimizer.py').read().split("# Candidate rule descriptors")[0])
N=len(O);TFS=[2,4,6,8,12,24];print('OOS23|n=%d'%N,flush=True)
H={}
for sym in sorted(set(x['sym'] for x in O)):
 xs=[x for x in O if x['sym']==sym];st=min(x['ts'] for x in xs)-pd.Timedelta(days=5);en=max(x['ts'] for x in xs)+pd.Timedelta(hours=1081)
 try:H[sym]=hourly(sym,int(st.timestamp()*1000),int(en.timestamp()*1000));print('H23|%s|%d'%(sym,len(H[sym])),flush=True)
 except Exception as e:H[sym]=pd.DataFrame();print('FAIL23|%s|%s'%(sym,str(e)[:100]),flush=True)

def atr(df,n=24):
 tr=pd.concat([df.high-df.low,(df.high-df.close.shift()).abs(),(df.low-df.close.shift()).abs()],axis=1).max(axis=1);return tr.rolling(n,min_periods=5).mean()
trades=[]
for ci,x in enumerate(O):
 p=H.get(x['sym'],pd.DataFrame());p=p[(p.index>=x['ts'])&(p.index<x['ts']+pd.Timedelta(hours=1081))]
 if p.empty or len(p)<24:continue
 # Do NOT truncate at legacy daily sell: exit tournament must see post-entry path independently.
 entry=float(x['entry']);oracle=float(p.close.max()/entry-1);A=atr(p);states={}
 for tf in TFS:
  z=bars(p,tf);bb,ss=psar_state(z);states[tf]=(z,bb,ss)
 trades.append(dict(ci=ci,sym=x['sym'],entry=entry,p=p,oracle=oracle,A=A,states=states))
print('TRADES23|%d'%len(trades),flush=True)

def first_bear(T,tf,pers=1):
 p=T['p'];z,bb,ss=T['states'][tf];cnt=0
 for j in range(6,len(p)):
  k=z.index.searchsorted(p.index[j],'right')-1;be=(k>=0 and not bb[k]);cnt=cnt+1 if be else 0
  if cnt>=pers:return j
 return len(p)-1

def policy(T,name):
 p=T['p'];entry=T['entry'];runhi=p.high.cummax();A=T['A']
 if name.startswith('SAR'):
  a=name[3:].split('P');tf=int(a[0]);pers=int(a[1]) if len(a)>1 else 1;return first_bear(T,tf,pers)
 if name.startswith('TRAIL'):
  q=float(name[5:])/100
  for j in range(12,len(p)):
   if p.close.iloc[j] <= runhi.iloc[j]*(1-q):return j
  return len(p)-1
 if name.startswith('ATR'):
  mult=float(name[3:])/10
  for j in range(24,len(p)):
   av=A.iloc[j]
   if np.isfinite(av) and p.close.iloc[j] <= runhi.iloc[j]-mult*av:return j
  return len(p)-1
 if name.startswith('BREADTH'):
  n=int(name[-1]);cnt=0
  for j in range(12,len(p)):
   b=0
   for tf in [4,6,8,12]:
    z,bb,_=T['states'][tf];k=z.index.searchsorted(p.index[j],'right')-1;b+=int(k>=0 and not bb[k])
   cnt=cnt+1 if b>=n else 0
   if cnt>=2:return j
  return len(p)-1
 if name.startswith('PROFIT'):
  # PROFIT{activation pct}_{trail pct}: activate trailing only after MFE threshold
  ac,tr=map(float,name[6:].split('_'));ac/=100;tr/=100
  for j in range(12,len(p)):
   mfe=runhi.iloc[j]/entry-1
   if mfe>=ac and p.close.iloc[j]<=runhi.iloc[j]*(1-tr):return j
  return len(p)-1
 return len(p)-1
names=[]
for tf in TFS:
 for pers in [1,2,3]:names.append('SAR%dP%d'%(tf,pers))
for q in [3,4,5,6,7,8,10,12,15]:names.append('TRAIL%d'%q)
for m in [15,20,25,30,35,40,50]:names.append('ATR%d'%m)
for n in [2,3,4]:names.append('BREADTH%d'%n)
for ac in [5,10,15,20,30]:
 for tr in [4,5,6,8,10,12]:names.append('PROFIT%d_%d'%(ac,tr))
R=[]
for T in trades:
 for name in names:
  j=policy(T,name);px=float(T['p'].close.iloc[j]);ret=px/T['entry']-1;oracle=T['oracle'];cap=ret/oracle if oracle>0 else np.nan;give=max(0,(T['p'].high.iloc[:j+1].max()-px)/T['p'].high.iloc[:j+1].max());R.append((T['ci'],T['sym'],name,ret,oracle,cap,give,j))
D=pd.DataFrame(R,columns=['ci','sym','policy','ret','oracle','capture','give','bars']);print('ROWS23|%d|policies=%d'%(len(D),len(names)),flush=True)
# Nested expanding walk-forward: choose policy on calibration solely by log growth, then freeze on next unseen block.
folds=[(430,430,600),(600,600,760),(760,760,920),(920,920,N)];wins=[]
for fi,(trainend,vstart,vend) in enumerate(folds,1):
 cal0=int(trainend*.65);cal=D[(D.ci>=cal0)&(D.ci<trainend)];scores=[]
 for name,q in cal.groupby('policy'):
  r=np.clip(q.ret.values,-.95,None);logg=np.mean(np.log1p(r));p10=np.quantile(r,.10);scores.append((logg,p10,np.median(r),name))
 scores.sort(reverse=True);best=scores[0][3];print('SELECT23|fold=%d|best=%s|cal_log=%.5f|cal_med=%.4f|cal_p10=%.4f'%(fi,best,scores[0][0],scores[0][2],scores[0][1]),flush=True)
 for rank,sc in enumerate(scores[:8],1):print('TOP23|fold=%d|rank=%d|%s|log=%.5f|med=%.4f|p10=%.4f'%(fi,rank,sc[3],sc[0],sc[2],sc[1]),flush=True)
 q=D[(D.ci>=vstart)&(D.ci<vend)&(D.policy==best)];r=np.clip(q.ret.values,-.95,None);logg=np.mean(np.log1p(r));print('TEST23|fold=%d|policy=%s|n=%d|log=%.5f|geo=%.4f|med=%.4f|mean=%.4f|p10=%.4f|capture=%.3f|give=%.3f'%(fi,best,len(q),logg,np.exp(logg)-1,np.median(r),np.mean(r),np.quantile(r,.10),np.nanmedian(q.capture),np.median(q.give)),flush=True);wins.append((fi,best,len(q),logg,np.exp(logg)-1,np.median(r),np.mean(r),np.quantile(r,.10),np.nanmedian(q.capture),np.median(q.give)))
# fixed-policy OOS tournament over all validation blocks combined, diagnostic only (not selection)
val=D[D.ci>=430];agg=[]
for name,q in val.groupby('policy'):
 r=np.clip(q.ret.values,-.95,None);agg.append((np.mean(np.log1p(r)),np.exp(np.mean(np.log1p(r)))-1,np.median(r),np.mean(r),np.quantile(r,.10),np.nanmedian(q.capture),np.median(q.give),name))
agg.sort(reverse=True)
for rank,a in enumerate(agg[:20],1):print('AGG23|rank=%d|%s|log=%.5f|geo=%.4f|med=%.4f|mean=%.4f|p10=%.4f|capture=%.3f|give=%.3f'%(rank,a[7],a[0],a[1],a[2],a[3],a[4],a[5],a[6]),flush=True)
W=pd.DataFrame(wins,columns=['fold','policy','n','log','geo','med','mean','p10','capture','give']);print('WF23|folds=%d|geo_med=%.4f|ret_med=%.4f|p10_med=%.4f|capture_med=%.3f|give_med=%.3f'%(len(W),W.geo.median(),W.med.median(),W.p10.median(),W.capture.median(),W.give.median()),flush=True)
print('DONE23',flush=True)