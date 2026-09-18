from itertools import combinations
import warnings
import numpy as np,pandas as pd
from crms import ASSETS,fetch,indicators,weekly
warnings.filterwarnings('ignore')

def raw(s):
 d=fetch(s);z=indicators(d).join(weekly(d));c=z.close
 for n in [3,5,10,20,50]:z[f'r{n}']=c.pct_change(n)
 for n in [20,50,200]:z[f'e{n}']=c.ewm(span=n,adjust=False).mean()
 z['stack']=(c>z.e20)&(z.e20>z.e50)&(z.e50>z.e200);z['adxup']=z.adx>z.adx.shift(3);z['dip']=z.plus_di-z.minus_di;z['rsiup']=z.rsi>z.rsi.shift(3);z['break20']=c>=z.high.shift().rolling(20).max();z['break50']=c>=z.high.shift().rolling(50).max();z['atrp']=z.atr/c;z['atrexp']=z.atrp>z.atrp.shift(5);z['psar_age']=z.psar_bull.groupby(z.psar_bull.ne(z.psar_bull.shift()).cumsum()).cumcount();z['psar_new3']=z.psar_bull&(z.psar_age<=2)
 for h in [7,10,15]:z[f'y{h}']=c.shift(-h)/c-1
 z['symbol']=s;return z
F=[]
for s in ASSETS:
 try:F.append(raw(s))
 except:pass
btc=next(x for x in F if x.symbol.iloc[0]=='BTCUSDT')
b=btc[['r5','r10','r20','e20','e50','e200','adx']].rename(columns={x:'b'+x for x in ['r5','r10','r20','e20','e50','e200','adx']})
for z in F:
 z[b.columns]=b.reindex(z.index).ffill();z['btcbull']=(z.be20>z.be50)&(z.be50>z.be200);z['btcmom']=z.br20>0;z['rs5']=z.r5-z.br5;z['rs10']=z.r10-z.br10;z['rs20']=z.r20-z.br20
df=pd.concat(F).sort_index();D=df.index.unique().sort_values();c1=D[int(len(D)*.55)];c2=D[int(len(D)*.75)];tr=df[df.index<c1];va=df[(df.index>=c1)&(df.index<c2)];te=df[df.index>=c2]
C={'weekly':lambda x:x.weekly_bull.eq(True),'stack':lambda x:x.stack,'btcbull':lambda x:x.btcbull,'btcmom':lambda x:x.btcmom,'macdpos':lambda x:x.macd_hist>0,'macdup':lambda x:x.macd_up,'dip':lambda x:x.dip>0,'di10':lambda x:x.dip>10,'adx20':lambda x:x.adx>=20,'adx25':lambda x:x.adx>=25,'adxup':lambda x:x.adxup,'rsi50':lambda x:x.rsi>=50,'rsi5570':lambda x:(x.rsi>=55)&(x.rsi<=70),'rsiup':lambda x:x.rsiup,'rv1':lambda x:x.rvol20>=1,'rv15':lambda x:x.rvol20>=1.5,'m5':lambda x:x.r5>.02,'m10':lambda x:x.r10>.05,'m20':lambda x:x.r20>.1,'break20':lambda x:x.break20,'break50':lambda x:x.break50,'rs5':lambda x:x.rs5>0,'rs10':lambda x:x.rs10>0,'rs20':lambda x:x.rs20>0,'rs10x':lambda x:x.rs10>.05,'atrexp':lambda x:x.atrexp,'psar':lambda x:x.psar_bull,'psarnew3':lambda x:x.psar_new3}
def ev(x,r,h):
 m=pd.Series(True,index=x.index,dtype=bool)
 for q in r:m&=C[q](x).fillna(False)
 y=x.loc[m,f'y{h}'].dropna();return len(y),float((y>0).mean()) if len(y) else 0,float(y.mean()) if len(y) else 0,float(y.median()) if len(y) else 0
# Rules are selected using TRAIN+VALIDATION only. TEST is never used for ranking.
for h in [7,10,15]:
 cand=[];N=list(C)
 for k in range(2,7):
  for r in combinations(N,k):
   n,w,mu,md=ev(tr,r,h)
   if n<100:continue
   nv,wv,mv,mdv=ev(va,r,h)
   if nv>=35:cand.append((min(w,wv),wv,n+nv,r))
 cand.sort(reverse=True)
 print(f'HORIZON|{h}|train_end={c1.date()}|validation_end={c2.date()}|test_start={c2.date()}')
 shown=0
 for score,wv,nv,r in cand:
  nt,wt,mt,mdt=ev(te,r,h)
  if nt<30:continue
  print('RULE|%s|pretest_score=%.4f|test_n=%d|test_win=%.4f|mean=%.4f|median=%.4f'%('+'.join(r),score,nt,wt,mt,mdt));shown+=1
  if shown>=20:break
