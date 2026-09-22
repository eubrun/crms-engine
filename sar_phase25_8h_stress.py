"""Phase25: robustness/stress test of fixed Daily-PSAR BUY -> first bearish SAR exit. Compare 4/6/8/12/24h, costs, eras, assets. No ML tuning."""
import numpy as np,pandas as pd
exec(open('sar_phase18c_optimizer.py').read().split("# Candidate rule descriptors")[0])
N=len(O);TFS=[4,6,8,12,24];print('OOS25|n=%d'%N,flush=True)
H={}
for sym in sorted(set(x['sym'] for x in O)):
 xs=[x for x in O if x['sym']==sym];st=min(x['ts'] for x in xs)-pd.Timedelta(days=5);en=max(x['ts'] for x in xs)+pd.Timedelta(hours=1081)
 try:H[sym]=hourly(sym,int(st.timestamp()*1000),int(en.timestamp()*1000));print('H25|%s|%d'%(sym,len(H[sym])),flush=True)
 except Exception as e:H[sym]=pd.DataFrame();print('FAIL25|%s|%s'%(sym,str(e)[:100]),flush=True)
R=[]
for ci,x in enumerate(O):
 p=H.get(x['sym'],pd.DataFrame());p=p[(p.index>=x['ts'])&(p.index<x['ts']+pd.Timedelta(hours=1081))]
 if p.empty or len(p)<24:continue
 entry=float(x['entry']);oracle=float(p.close.max()/entry-1)
 for tf in TFS:
  z=bars(p,tf);bb,ss=psar_state(z);j=None
  for h in range(6,len(p)):
   k=z.index.searchsorted(p.index[h],'right')-1
   if k>=0 and not bb[k]:j=h;break
  if j is None:j=len(p)-1
  px=float(p.close.iloc[j]);ret=px/entry-1;runhi=float(p.high.iloc[:j+1].max());give=max(0,1-px/runhi);cap=ret/oracle if oracle>0 else np.nan
  R.append((ci,x['sym'],x['ts'],tf,ret,give,cap,j/24))
D=pd.DataFrame(R,columns=['ci','sym','ts','tf','ret','give','capture','days']);print('ROWS25|%d|trades=%d'%(len(D),D.ci.nunique()),flush=True)
# chronological OOS blocks already used in Phase23
blocks=[(430,600),(600,760),(760,920),(920,N)]
for tf in TFS:
 q=D[(D.ci>=430)&(D.tf==tf)].copy();r=np.clip(q.ret.values,-.95,None);lg=np.mean(np.log1p(r));print('ALL25|tf=%d|n=%d|geo=%.4f|med=%.4f|mean=%.4f|p10=%.4f|give=%.4f|capture=%.3f'%(tf,len(q),np.exp(lg)-1,np.median(r),np.mean(r),np.quantile(r,.1),np.median(q.give),np.nanmedian(q.capture)),flush=True)
 for bi,(a,b) in enumerate(blocks,1):
  w=D[(D.ci>=a)&(D.ci<b)&(D.tf==tf)];rr=np.clip(w.ret.values,-.95,None)
  if len(rr):print('BLOCK25|tf=%d|b=%d|n=%d|geo=%.4f|med=%.4f|p10=%.4f|give=%.4f'%(tf,bi,len(rr),np.exp(np.mean(np.log1p(rr)))-1,np.median(rr),np.quantile(rr,.1),np.median(w.give)),flush=True)
# transaction cost/slippage round trip stress, deducted from raw trade return multiplicatively approx
for costbp in [10,25,50,75,100,150,200]:
 vals=[]
 for tf in TFS:
  q=D[(D.ci>=430)&(D.tf==tf)];net=(1+q.ret.values)*(1-costbp/10000)-1;geo=np.exp(np.mean(np.log1p(np.clip(net,-.95,None))))-1;vals.append((geo,tf))
 vals.sort(reverse=True);print('COST25|bp=%d|winner=%dh|geo=%.4f|8h=%.4f'%(costbp,vals[0][1],vals[0][0],[v for v,t in vals if t==8][0]),flush=True)
# per-asset: compare 8h with nearest rivals, require adequate sample
for sym,g in D[(D.ci>=430)&(D.tf.isin([6,8,12]))].groupby('sym'):
 piv=g.pivot(index='ci',columns='tf',values='ret');
 if len(piv)<10:continue
 out=[]
 for tf in [6,8,12]:
  if tf in piv:out.append('%dh=%.3f'%(tf,np.exp(np.mean(np.log1p(np.clip(piv[tf].dropna().values,-.95,None))))-1))
 print('ASSET25|%s|n=%d|%s'%(sym,len(piv),'|'.join(out)),flush=True)
# bootstrap paired 8h vs 6h and 12h on OOS trades; probability mean log advantage >0
rng=np.random.default_rng(2501);P=D[D.ci>=430].pivot(index='ci',columns='tf',values='ret')
for rival in [4,6,12,24]:
 x=P[[8,rival]].dropna();d=np.log1p(np.clip(x[8].values,-.95,None))-np.log1p(np.clip(x[rival].values,-.95,None));boots=[]
 for _ in range(5000):boots.append(np.mean(rng.choice(d,len(d),replace=True)))
 lo,hi=np.quantile(boots,[.025,.975]);print('BOOT25|8v%d|n=%d|dlog=%.5f|pgt0=%.3f|ci95=[%.5f,%.5f]'%(rival,len(d),np.mean(d),np.mean(np.array(boots)>0),lo,hi),flush=True)
print('DONE25',flush=True)