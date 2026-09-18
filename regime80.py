from itertools import combinations
import warnings
import numpy as np,pandas as pd
from crms import ASSETS,fetch,indicators,weekly
warnings.filterwarnings('ignore')
def raw(s):
 d=fetch(s);z=indicators(d).join(weekly(d));c=z.close
 for n in [3,5,10,20,50]:z[f'r{n}']=c.pct_change(n)
 for n in [20,50,200]:z[f'e{n}']=c.ewm(span=n,adjust=False).mean()
 z['ema_stack']=(c>z.e20)&(z.e20>z.e50)&(z.e50>z.e200);z['adxup']=z.adx>z.adx.shift(3);z['dip']=z.plus_di-z.minus_di;z['rsiup']=z.rsi>z.rsi.shift(3);z['break20']=c>=z.high.shift().rolling(20).max();z['break50']=c>=z.high.shift().rolling(50).max();z['atrp']=z.atr/c;z['atrexp']=z.atrp>z.atrp.shift(5);z['psar_age']=z.psar_bull.groupby(z.psar_bull.ne(z.psar_bull.shift()).cumsum()).cumcount();z['psar_new3']=z.psar_bull&(z.psar_age<=2)
 for h in [7,10,15]:z[f'y{h}']=c.shift(-h)/c-1
 z['symbol']=s;return z
F=[]
for s in ASSETS:
 try:F.append(raw(s));print('LOAD|'+s,flush=True)
 except Exception as e:print('FAIL|'+s+'|'+str(e)[:100],flush=True)
btc=next(x for x in F if x.symbol.iloc[0]=='BTCUSDT');b=btc[['r5','r10','r20','e20','e50','e200']].rename(columns={x:'b'+x for x in ['r5','r10','r20','e20','e50','e200']})
for z in F:
 z[b.columns]=b.reindex(z.index).ffill();z['btcbull']=(z.be20>z.be50)&(z.be50>z.be200);z['btcmom']=z.br20>0;z['rs5']=z.r5-z.br5;z['rs10']=z.r10-z.br10;z['rs20']=z.r20-z.br20
df=pd.concat(F).sort_index();D=pd.DatetimeIndex(pd.to_datetime(df.index,utc=True)).tz_convert(None).unique().sort_values();c1=D[int(len(D)*.55)];c2=D[int(len(D)*.75)]
C={'weekly':df.weekly_bull.eq(True),'stack':df.ema_stack,'btcbull':df.btcbull,'btcmom':df.btcmom,'macdpos':df.macd_hist>0,'macdup':df.macd_up,'dip':df.dip>0,'di10':df.dip>10,'adx20':df.adx>=20,'adx25':df.adx>=25,'adxup':df.adxup,'rsi50':df.rsi>=50,'rsi5570':(df.rsi>=55)&(df.rsi<=70),'rsiup':df.rsiup,'rv1':df.rvol20>=1,'rv15':df.rvol20>=1.5,'m5':df.r5>.02,'m10':df.r10>.05,'m20':df.r20>.1,'break20':df.break20,'break50':df.break50,'rs5':df.rs5>0,'rs10':df.rs10>0,'rs20':df.rs20>0,'rs10x':df.rs10>.05,'atrexp':df.atrexp,'psar':df.psar_bull,'psarnew3':df.psar_new3}
X=np.column_stack([C[k].fillna(False).to_numpy(bool) for k in C]);names=list(C)
idx=pd.DatetimeIndex(pd.to_datetime(df.index,utc=True)).tz_convert(None).to_numpy(dtype='datetime64[ns]');c1n=np.datetime64(c1.to_datetime64(),'ns');c2n=np.datetime64(c2.to_datetime64(),'ns');train=idx<c1n;valid=(idx>=c1n)&(idx<c2n);test=idx>=c2n
def stat(mask,y,part):
 q=mask&part&np.isfinite(y);n=int(q.sum());return n,float((y[q]>0).mean()) if n else 0,float(y[q].mean()) if n else 0,float(np.median(y[q])) if n else 0
print(f'SPLIT|{c1.date()}|{c2.date()}|assets={len(F)}|rows={len(df)}',flush=True)
for h in [7,10,15]:
 print(f'START|h={h}',flush=True);y=df[f'y{h}'].to_numpy(float);cand=[];checked=0
 for k in range(2,7):
  for ids in combinations(range(len(names)),k):
   checked+=1;m=X[:,ids].all(axis=1);n,w,_,_=stat(m,y,train)
   if n<100:continue
   nv,wv,_,_=stat(m,y,valid)
   if nv>=35:cand.append((min(w,wv),wv,n+nv,ids))
  print(f'PROGRESS|h={h}|k={k}|checked={checked}|candidates={len(cand)}',flush=True)
 cand.sort(reverse=True);shown=0
 for score,wv,nv,ids in cand:
  m=X[:,ids].all(axis=1);nt,wt,mt,mdt=stat(m,y,test)
  if nt<30:continue
  print('RULE|h=%d|%s|pre=%.4f|test_n=%d|win=%.4f|mean=%.4f|median=%.4f'%(h,'+'.join(names[i] for i in ids),score,nt,wt,mt,mdt),flush=True);shown+=1
  if shown>=20:break
 print(f'DONE|h={h}|checked={checked}',flush=True)
