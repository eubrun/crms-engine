from itertools import combinations
import warnings,numpy as np,pandas as pd
from crms import ASSETS,fetch,indicators,weekly
warnings.filterwarnings('ignore'); rows=[]
for s in ASSETS:
 try:
  z=indicators(fetch(s)).join(weekly(fetch(s)));c=z.close
  z['atrp']=z.atr/c;z['hist']=z.macd_hist;z['dip']=z.plus_di-z.minus_di
  bull=z.psar_bull.astype(bool);en=bull&~bull.shift(1,fill_value=False);ex=~bull&bull.shift(1,fill_value=False);E=np.flatnonzero(en.to_numpy());X=np.flatnonzero(ex.to_numpy())
  # bearish SAR age immediately before bullish flip
  bear=(~bull); grp=bear.ne(bear.shift()).cumsum();age=bear.groupby(grp).cumcount()+1
  for i in E:
   q=[j for j in X if j>i]
   if not q or i<10:continue
   j=q[0];r={'symbol':s,'ret':float(c.iloc[j]/c.iloc[i]-1),'days':j-i,'date':z.index[i]}
   # only information known at entry, mostly slopes/compression in prior 3-10d
   r.update({'bear5':age.iloc[i-1]>=5,'bear10':age.iloc[i-1]>=10,'adxlow20':z.adx.iloc[i-1]<20,'adxlow25':z.adx.iloc[i-1]<25,'adxturn':z.adx.iloc[i-1]>z.adx.iloc[i-4],'dicontract':z.dip.iloc[i-1]>z.dip.iloc[i-4],'minusfall':z.minus_di.iloc[i-1]<z.minus_di.iloc[i-4],'histimprove':z['hist'].iloc[i-1]>z['hist'].iloc[i-4],'histimprove7':z['hist'].iloc[i-1]>z['hist'].iloc[i-8],'rsirise':z.rsi.iloc[i-1]>z.rsi.iloc[i-4],'rsiunder55':z.rsi.iloc[i-1]<55,'volwake':z.rvol20.iloc[i-1]>z.rvol20.iloc[i-4],'volquiet':z.rvol20.iloc[i-1]<1.5,'atrcompress':z.atrp.iloc[i-1]<z.atrp.iloc[i-6:i].mean(),'atrexpand':z.atrp.iloc[i-1]>z.atrp.iloc[i-4],'price5pos':c.iloc[i-1]>c.iloc[i-6],'price10pos':c.iloc[i-1]>c.iloc[i-11],'weekly':bool(z.weekly_bull.iloc[i]) if pd.notna(z.weekly_bull.iloc[i]) else False})
   rows.append(r)
 except Exception as e:print('FAIL|%s|%s'%(s,str(e)[:80]),flush=True)
t=pd.DataFrame(rows).set_index('date').sort_index(); names=[x for x in t.columns if x not in ['symbol','ret','days']];A=t[names].fillna(False).to_numpy(bool);y=t.ret.to_numpy();D=pd.DatetimeIndex(pd.to_datetime(t.index,utc=True)).tz_convert(None).unique().sort_values();c1=D[int(len(D)*.55)];c2=D[int(len(D)*.75)];idx=pd.DatetimeIndex(pd.to_datetime(t.index,utc=True)).tz_convert(None);P={'tr':idx<c1,'va':(idx>=c1)&(idx<c2),'te':idx>=c2}
def st(m,p):
 q=m&np.asarray(P[p]);n=q.sum();return int(n),float((y[q]>0).mean()) if n else 0,float(y[q].mean()) if n else 0,float(np.median(y[q])) if n else 0
print('BASE|n=%d|win=%.4f|mean=%.4f|days=%.2f|split=%s,%s'%(len(t),(y>0).mean(),y.mean(),t.days.mean(),c1.date(),c2.date()),flush=True);cand=[];checked=0
for k in range(1,7):
 for ids in combinations(range(len(names)),k):
  checked+=1;m=A[:,ids].all(1);n,w,_,_=st(m,'tr');nv,wv,_,_=st(m,'va')
  if n>=60 and nv>=20:cand.append((min(w,wv),ids,w,wv))
 print('PROGRESS|k=%d|checked=%d|cand=%d'%(k,checked,len(cand)),flush=True)
cand.sort(reverse=True);shown=0
for pre,ids,w,wv in cand:
 m=A[:,ids].all(1);nt,wt,mu,md=st(m,'te')
 if nt<15:continue
 print('RULE|%s|train=%.4f|valid=%.4f|test_n=%d|test=%.4f|mean=%.4f|median=%.4f'%('+'.join(names[i] for i in ids),w,wv,nt,wt,mu,md),flush=True);shown+=1
 if shown>=40:break
print('DONE|checked=%d'%checked,flush=True)
