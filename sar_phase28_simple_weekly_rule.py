"""Phase28: anti-overfit validation of simple Weekly+ exit thresholds. At first 8H bearish: low profit->8H, medium->12H, high->Daily. Thresholds selected on prior history only."""
import numpy as np,pandas as pd
exec(open('sar_phase18c_optimizer.py').read().split("# Candidate rule descriptors")[0])
N=len(O);print('OOS28|n=%d'%N,flush=True)
H={}
for sym in sorted(set(x['sym'] for x in O)):
 xs=[x for x in O if x['sym']==sym];st=min(x['ts'] for x in xs)-pd.Timedelta(days=180);en=max(x['ts'] for x in xs)+pd.Timedelta(hours=1081)
 try:H[sym]=hourly(sym,int(st.timestamp()*1000),int(en.timestamp()*1000));print('H28|%s|%d'%(sym,len(H[sym])),flush=True)
 except Exception as e:H[sym]=pd.DataFrame();print('FAIL28|%s|%s'%(sym,str(e)[:100]),flush=True)
def firstbear(p,tf):
 z=bars(p,tf);b,s=psar_state(z)
 for j in range(6,len(p)):
  k=z.index.searchsorted(p.index[j],'right')-1
  if k>=0 and not b[k]:return j
 return len(p)-1
R=[]
for ci,x in enumerate(O):
 full=H.get(x['sym'],pd.DataFrame());hist=full[full.index<=x['ts']];zw=bars(hist,168)
 if len(zw)<5:continue
 bw,sw=psar_state(zw);kw=zw.index.searchsorted(x['ts'],'right')-1
 if kw<0:continue
 wb=int(bool(bw[kw]));p=full[(full.index>=x['ts'])&(full.index<x['ts']+pd.Timedelta(hours=1081))]
 if len(p)<48:continue
 entry=float(x['entry']);j8=firstbear(p,8);j12=firstbear(p,12);j24=firstbear(p,24);r8=float(p.close.iloc[j8]/entry-1);r12=float(p.close.iloc[j12]/entry-1);r24=float(p.close.iloc[j24]/entry-1)
 R.append((ci,x['sym'],wb,r8,r12,r24))
D=pd.DataFrame(R,columns=['ci','sym','wb','r8','r12','r24']);print('ROWS28|%d|wbull=%d|wbear=%d'%(len(D),(D.wb==1).sum(),(D.wb==0).sum()),flush=True)
# candidate low/high profit thresholds; rule: r8<lo => sell8; lo<=r8<hi => sell12; r8>=hi => sell24
los=[0,.01,.02,.03,.04,.05];his=[.05,.06,.08,.10,.12,.15]
folds=[(430,430,600),(600,600,760),(760,760,920),(920,920,N)];outs=[]
for fi,(te,va,vb) in enumerate(folds,1):
 cal=D[(D.ci>=int(te*.55))&(D.ci<te)&(D.wb==1)];best=None
 for lo in los:
  for hi in his:
   if hi<=lo:continue
   rr=np.where(cal.r8<lo,cal.r8,np.where(cal.r8<hi,cal.r12,cal.r24));lg=np.mean(np.log1p(np.clip(rr,-.95,None))) if len(rr) else -99
   # small complexity/instability penalty favors broad thresholds, but objective remains PnL
   if best is None or lg>best[0]:best=(lg,lo,hi)
 lo,hi=best[1],best[2];q=D[(D.ci>=va)&(D.ci<vb)&(D.wb==1)];base=np.clip(q.r8.values,-.95,None);rule=np.clip(np.where(q.r8<lo,q.r8,np.where(q.r8<hi,q.r12,q.r24)),-.95,None);bl=np.mean(np.log1p(base));rl=np.mean(np.log1p(rule));acts=np.where(q.r8<lo,8,np.where(q.r8<hi,12,24));print('FOLD28|f=%d|lo=%.3f|hi=%.3f|n=%d|basegeo=%.4f|rulegeo=%.4f|dlog=%.5f|p10base=%.4f|p10rule=%.4f|a8=%.3f|a12=%.3f|a24=%.3f'%(fi,lo,hi,len(q),np.exp(bl)-1,np.exp(rl)-1,rl-bl,np.quantile(base,.1) if len(base) else 0,np.quantile(rule,.1) if len(rule) else 0,np.mean(acts==8) if len(acts) else 0,np.mean(acts==12) if len(acts) else 0,np.mean(acts==24) if len(acts) else 0),flush=True);outs.append((fi,lo,hi,np.exp(bl)-1,np.exp(rl)-1,rl-bl,np.quantile(base,.1),np.quantile(rule,.1)))
A=pd.DataFrame(outs,columns=['f','lo','hi','base','rule','dlog','bp10','rp10']);print('AGG28|wins=%d/4|basegeo_med=%.4f|rulegeo_med=%.4f|dlog_med=%.5f|lo_med=%.3f|hi_med=%.3f|bp10_med=%.4f|rp10_med=%.4f'%((A.dlog>0).sum(),A.base.median(),A.rule.median(),A.dlog.median(),A.lo.median(),A.hi.median(),A.bp10.median(),A.rp10.median()),flush=True)
# fixed 3/8 thresholds diagnostic on unseen blocks (predeclared from Phase27, not selected here)
for fi,(te,va,vb) in enumerate(folds,1):
 q=D[(D.ci>=va)&(D.ci<vb)&(D.wb==1)];b=np.clip(q.r8.values,-.95,None);r=np.clip(np.where(q.r8<.03,q.r8,np.where(q.r8<.08,q.r12,q.r24)),-.95,None);print('FIXED28|f=%d|n=%d|basegeo=%.4f|rule38geo=%.4f|dlog=%.5f|p10=%.4f'%(fi,len(q),np.exp(np.mean(np.log1p(b)))-1,np.exp(np.mean(np.log1p(r)))-1,np.mean(np.log1p(r))-np.mean(np.log1p(b)),np.quantile(r,.1) if len(r) else 0),flush=True)
# combined system diagnostic: Weekly- always 8H; Weekly+ fixed 3/8 rule
for fi,(te,va,vb) in enumerate(folds,1):
 q=D[(D.ci>=va)&(D.ci<vb)];base=np.clip(q.r8.values,-.95,None);rule=np.where(q.wb==0,q.r8,np.where(q.r8<.03,q.r8,np.where(q.r8<.08,q.r12,q.r24)));rule=np.clip(rule,-.95,None);print('SYSTEM28|f=%d|n=%d|base8geo=%.4f|hiergeo=%.4f|dlog=%.5f|basep10=%.4f|hierp10=%.4f'%(fi,len(q),np.exp(np.mean(np.log1p(base)))-1,np.exp(np.mean(np.log1p(rule)))-1,np.mean(np.log1p(rule))-np.mean(np.log1p(base)),np.quantile(base,.1),np.quantile(rule,.1)),flush=True)
print('DONE28',flush=True)