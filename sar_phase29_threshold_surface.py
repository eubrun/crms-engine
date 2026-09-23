"""Phase29: robustness surface for Weekly+ hierarchical exit. Weekly- always 8H. Weekly+ at first 8H bearish: r8<lo -> 8H; lo<=r8<hi -> 12H; >=hi -> Daily. Test grid, folds, bootstrap neighborhood."""
import numpy as np,pandas as pd
exec(open('sar_phase18c_optimizer.py').read().split("# Candidate rule descriptors")[0])
N=len(O);print('OOS29|n=%d'%N,flush=True)
H={}
for sym in sorted(set(x['sym'] for x in O)):
 xs=[x for x in O if x['sym']==sym];st=min(x['ts'] for x in xs)-pd.Timedelta(days=180);en=max(x['ts'] for x in xs)+pd.Timedelta(hours=1081)
 try:H[sym]=hourly(sym,int(st.timestamp()*1000),int(en.timestamp()*1000));print('H29|%s|%d'%(sym,len(H[sym])),flush=True)
 except Exception as e:H[sym]=pd.DataFrame();print('FAIL29|%s|%s'%(sym,str(e)[:100]),flush=True)
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
 e=float(x['entry']);j8=firstbear(p,8);j12=firstbear(p,12);j24=firstbear(p,24);R.append((ci,x['sym'],wb,float(p.close.iloc[j8]/e-1),float(p.close.iloc[j12]/e-1),float(p.close.iloc[j24]/e-1)))
D=pd.DataFrame(R,columns=['ci','sym','wb','r8','r12','r24']);D=D[D.ci>=430].copy();print('ROWS29|%d|wbull=%d|wbear=%d'%(len(D),(D.wb==1).sum(),(D.wb==0).sum()),flush=True)
los=[.02,.025,.03,.035,.04,.045,.05];his=[.08,.09,.10,.11,.12];folds=[(430,600),(600,760),(760,920),(920,N)];surface=[]
for lo in los:
 for hi in his:
  allr=[];allb=[];wins=0;foldvals=[]
  for fi,(a,b) in enumerate(folds,1):
   q=D[(D.ci>=a)&(D.ci<b)];base=np.clip(q.r8.values,-.95,None);rule=np.where(q.wb==0,q.r8,np.where(q.r8<lo,q.r8,np.where(q.r8<hi,q.r12,q.r24)));rule=np.clip(rule,-.95,None);bl=np.mean(np.log1p(base));rl=np.mean(np.log1p(rule));wins+=rl>bl;foldvals.append(rl-bl);allr.extend(rule);allb.extend(base)
  d=np.mean(np.log1p(allr))-np.mean(np.log1p(allb));surface.append((d,wins,lo,hi,np.median(foldvals),min(foldvals),np.exp(np.mean(np.log1p(allr)))-1))
surface.sort(reverse=True);print('TOP29|'+ '|'.join(['lo=%.3f,hi=%.3f,dlog=%.5f,w=%d,med=%.5f,min=%.5f,geo=%.4f'%(x[2],x[3],x[0],x[1],x[4],x[5],x[6]) for x in surface[:15]]),flush=True)
# neighborhood summary: how much of grid improves baseline and 3+/4 folds
arr=np.array([[x[0],x[1],x[2],x[3],x[4],x[5],x[6]] for x in surface]);print('ROBUST29|grid=%d|positive=%d|positive_pct=%.3f|wins3plus=%d|wins3plus_pct=%.3f|dlog_med=%.5f|dlog_p25=%.5f|dlog_p75=%.5f'%(len(arr),int((arr[:,0]>0).sum()),np.mean(arr[:,0]>0),int((arr[:,1]>=3).sum()),np.mean(arr[:,1]>=3),np.median(arr[:,0]),np.quantile(arr[:,0],.25),np.quantile(arr[:,0],.75)),flush=True)
# print every grid cell compactly
for lo in los:
 z=[x for x in surface if abs(x[2]-lo)<1e-9];z=sorted(z,key=lambda x:x[3]);print('GRID29|lo=%.3f|%s'%(lo,'|'.join(['hi%.2f:d%.4f:w%d'%(x[3],x[0],x[1]) for x in z])),flush=True)
# paired bootstrap for fixed 3/8 and robust-center 3.5/10 across all trades
rng=np.random.default_rng(2901)
for lo,hi,name in [(.03,.08,'3_8'),(.035,.10,'3.5_10'),(.04,.10,'4_10')]:
 q=D;base=np.clip(q.r8.values,-.95,None);rule=np.where(q.wb==0,q.r8,np.where(q.r8<lo,q.r8,np.where(q.r8<hi,q.r12,q.r24)));rule=np.clip(rule,-.95,None);delta=np.log1p(rule)-np.log1p(base);boots=np.array([np.mean(rng.choice(delta,len(delta),replace=True)) for _ in range(5000)]);lo95,hi95=np.quantile(boots,[.025,.975]);print('BOOT29|%s|dlog=%.5f|pgt0=%.3f|ci95=[%.5f,%.5f]|geo=%.4f|basegeo=%.4f'%(name,delta.mean(),np.mean(boots>0),lo95,hi95,np.exp(np.mean(np.log1p(rule)))-1,np.exp(np.mean(np.log1p(base)))-1),flush=True)
print('DONE29',flush=True)