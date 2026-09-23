"""Phase26: Does Weekly PSAR state at fixed Daily-PSAR BUY change optimal exit? Split entries by causal Weekly PSAR bullish/bearish and compare first bearish 4/6/8/12/24h exits. Weekly bars use completed 7d bars only."""
import numpy as np,pandas as pd
exec(open('sar_phase18c_optimizer.py').read().split("# Candidate rule descriptors")[0])
N=len(O);TFS=[4,6,8,12,24];print('OOS26|n=%d'%N,flush=True)
H={}
for sym in sorted(set(x['sym'] for x in O)):
 xs=[x for x in O if x['sym']==sym];st=min(x['ts'] for x in xs)-pd.Timedelta(days=180);en=max(x['ts'] for x in xs)+pd.Timedelta(hours=1081)
 try:H[sym]=hourly(sym,int(st.timestamp()*1000),int(en.timestamp()*1000));print('H26|%s|%d'%(sym,len(H[sym])),flush=True)
 except Exception as e:H[sym]=pd.DataFrame();print('FAIL26|%s|%s'%(sym,str(e)[:100]),flush=True)
R=[]
for ci,x in enumerate(O):
 full=H.get(x['sym'],pd.DataFrame());p=full[(full.index>=x['ts'])&(full.index<x['ts']+pd.Timedelta(hours=1081))]
 if p.empty or len(p)<24:continue
 entry=float(x['entry'])
 # Weekly context strictly from bars whose 7d period is completed before entry.
 # Reuse generic bars with 168h; search right then require timestamp <= entry. PSAR is computed only on history through entry.
 hist=full[full.index<=x['ts']]
 zw=bars(hist,168)
 if len(zw)<5:continue
 bw,sw=psar_state(zw);kw=zw.index.searchsorted(x['ts'],'right')-1
 if kw<0:continue
 weekly=int(bool(bw[kw]));wdist=(entry-float(sw[kw]))/entry if np.isfinite(sw[kw]) else np.nan
 for tf in TFS:
  z=bars(p,tf);bb,ss=psar_state(z);j=None
  for h in range(6,len(p)):
   k=z.index.searchsorted(p.index[h],'right')-1
   if k>=0 and not bb[k]:j=h;break
  if j is None:j=len(p)-1
  px=float(p.close.iloc[j]);ret=px/entry-1;runhi=float(p.high.iloc[:j+1].max());give=max(0,1-px/runhi)
  R.append((ci,x['sym'],x['ts'],weekly,wdist,tf,ret,give,j/24))
D=pd.DataFrame(R,columns=['ci','sym','ts','weekly','wdist','tf','ret','give','days']).replace([np.inf,-np.inf],np.nan);print('ROWS26|%d|trades=%d|weeklyBull=%d|weeklyBear=%d'%(len(D),D.ci.nunique(),D[D.tf==8].weekly.sum(),(D[D.tf==8].weekly==0).sum()),flush=True)
for ws in [0,1]:
 lab='WBULL' if ws else 'WBEAR';q0=D[(D.ci>=430)&(D.weekly==ws)];print('GROUP26|%s|trades=%d'%(lab,q0.ci.nunique()),flush=True)
 vals=[]
 for tf in TFS:
  q=q0[q0.tf==tf];r=np.clip(q.ret.values,-.95,None)
  if not len(r):continue
  geo=np.exp(np.mean(np.log1p(r)))-1;vals.append((geo,tf));print('EXIT26|%s|tf=%d|n=%d|geo=%.4f|med=%.4f|mean=%.4f|p10=%.4f|give=%.4f|days=%.2f'%(lab,tf,len(r),geo,np.median(r),np.mean(r),np.quantile(r,.1),np.median(q.give),np.median(q.days)),flush=True)
 vals.sort(reverse=True);print('WIN26|%s|tf=%d|geo=%.4f'%(lab,vals[0][1],vals[0][0]),flush=True)
 # blocks to see stability of 8h vs group winner
 for bi,(a,b) in enumerate([(430,600),(600,760),(760,920),(920,N)],1):
  for tf in sorted(set([8,vals[0][1]])):
   q=D[(D.ci>=a)&(D.ci<b)&(D.weekly==ws)&(D.tf==tf)];r=np.clip(q.ret.values,-.95,None)
   if len(r):print('BLOCK26|%s|b=%d|tf=%d|n=%d|geo=%.4f'%(lab,bi,tf,len(r),np.exp(np.mean(np.log1p(r)))-1),flush=True)
# paired bootstrap: within weekly state compare 8h to alternatives
rng=np.random.default_rng(2601)
for ws in [0,1]:
 lab='WBULL' if ws else 'WBEAR';P=D[(D.ci>=430)&(D.weekly==ws)].pivot(index='ci',columns='tf',values='ret')
 for rival in [4,6,12,24]:
  x=P[[8,rival]].dropna();
  if len(x)<10:continue
  d=np.log1p(np.clip(x[8].values,-.95,None))-np.log1p(np.clip(x[rival].values,-.95,None));boots=np.array([np.mean(rng.choice(d,len(d),replace=True)) for _ in range(4000)]);lo,hi=np.quantile(boots,[.025,.975]);print('BOOT26|%s|8v%d|n=%d|dlog=%.5f|pgt0=%.3f|ci95=[%.5f,%.5f]'%(lab,rival,len(d),np.mean(d),np.mean(boots>0),lo,hi),flush=True)
# weekly distance quartiles with 8h exit: test whether strength above/below weekly SAR changes edge
q=D[(D.ci>=430)&(D.tf==8)&D.wdist.notna()].copy();q['quart']=pd.qcut(q.wdist,4,labels=False,duplicates='drop')
for z,g in q.groupby('quart'):
 r=np.clip(g.ret.values,-.95,None);print('WDIST26|q=%d|n=%d|wdist=%.4f|geo8=%.4f|med=%.4f'%(z,len(r),g.wdist.median(),np.exp(np.mean(np.log1p(r)))-1,np.median(r)),flush=True)
print('DONE26',flush=True)