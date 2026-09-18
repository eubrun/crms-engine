from itertools import combinations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from crms import ASSETS,fetch,indicators,weekly
warnings.filterwarnings('ignore',category=FutureWarning)
OUT=Path('output');OUT.mkdir(exist_ok=True)
def raw(symbol):
 d=fetch(symbol);z=indicators(d).join(weekly(d));c=z.close
 z['ret3']=c.pct_change(3);z['ret5']=c.pct_change(5);z['ret10']=c.pct_change(10);z['ret20']=c.pct_change(20)
 z['ema20']=c.ewm(span=20,adjust=False).mean();z['ema50']=c.ewm(span=50,adjust=False).mean();z['ema200']=c.ewm(span=200,adjust=False).mean();z['ema_stack']=(c>z.ema20)&(z.ema20>z.ema50)&(z.ema50>z.ema200)
 z['break20']=c>=z.high.shift(1).rolling(20).max();z['adx_up']=z.adx>z.adx.shift(3);z['di_spread']=z.plus_di-z.minus_di;z['rsi_up']=z.rsi>z.rsi.shift(3);z['atr_pct']=z.atr/c;z['atr_expand']=z.atr_pct>z.atr_pct.shift(5);z['y10']=c.shift(-10)/c-1;z['symbol']=symbol;return z
frames=[];fail={}
for s in ASSETS:
 try:frames.append(raw(s))
 except Exception as e:fail[s]=str(e)
btc=next(x for x in frames if x.symbol.iloc[0]=='BTCUSDT')[['ret5','ret10','ret20','ema_stack']].rename(columns={'ret5':'btc5','ret10':'btc10','ret20':'btc20','ema_stack':'btc_ema_stack'})
for z in frames:
 z[['btc5','btc10','btc20','btc_ema_stack']]=btc.reindex(z.index).ffill();z['rs5']=z.ret5-z.btc5;z['rs10']=z.ret10-z.btc10;z['rs20']=z.ret20-z.btc20
df=pd.concat(frames).sort_index();dates=df.index.unique().sort_values();cut=dates[int(len(dates)*.70)];train=df[df.index<cut].copy();test=df[df.index>=cut].copy()
conds={'weekly_bull':lambda x:x.weekly_bull.eq(True),'ema_stack':lambda x:x.ema_stack,'btc_bull':lambda x:x.btc_ema_stack.eq(True),'macd_pos':lambda x:x.macd_hist>0,'macd_up':lambda x:x.macd_up,'di_pos':lambda x:x.plus_di>x.minus_di,'di10':lambda x:x.di_spread>10,'adx20':lambda x:x.adx>=20,'adx25':lambda x:x.adx>=25,'adx_up':lambda x:x.adx_up,'rsi50_70':lambda x:(x.rsi>=50)&(x.rsi<=70),'rsi_up':lambda x:x.rsi_up,'rvol1':lambda x:x.rvol20>=1,'rvol1_5':lambda x:x.rvol20>=1.5,'mom5_2':lambda x:x.ret5>.02,'mom10_5':lambda x:x.ret10>.05,'mom20_10':lambda x:x.ret20>.10,'break20':lambda x:x.break20,'psar':lambda x:x.psar_bull,'rs5pos':lambda x:x.rs5>0,'rs10pos':lambda x:x.rs10>0,'rs20pos':lambda x:x.rs20>0,'rs10_5':lambda x:x.rs10>.05,'atr_expand':lambda x:x.atr_expand}
def ev(x,rule):
 m=pd.Series(True,index=x.index,dtype=bool)
 for n in rule:m &= conds[n](x).fillna(False).astype(bool)
 y=x.loc[m,'y10'].dropna();return len(y),(y>0).mean() if len(y) else np.nan,y.mean() if len(y) else np.nan,y.median() if len(y) else np.nan
c=[];names=list(conds)
for k in range(1,6):
 for rule in combinations(names,k):
  n,w,mean,med=ev(train,rule)
  if n>=80:c.append((w,n,mean,med,rule))
c.sort(reverse=True,key=lambda q:(q[0],q[1]));rows=[]
for w,n,mean,med,rule in c[:500]:
 nt,wt,mt,mdt=ev(test,rule)
 if nt>=30:rows.append({'rule':' + '.join(rule),'train_n':n,'train_win10':w,'test_n':nt,'test_win10':wt,'test_mean10':mt,'test_median10':mdt})
out=pd.DataFrame(rows).sort_values(['test_win10','test_n'],ascending=False);out.to_csv(OUT/'search80_expanded.csv',index=False)
print('RESULT|assets_ok=%d|assets_fail=%d|start=%s|end=%s|split=%s'%(len(frames),len(fail),df.index.min().date(),df.index.max().date(),cut.date()))
print('FAILURES|'+json.dumps(fail) if False else 'FAILURES|'+str(fail))
for _,r in out.head(15).iterrows():print('TOP|%s|train_n=%d|train_win=%.4f|test_n=%d|test_win=%.4f|mean=%.4f|median=%.4f'%(r.rule,r.train_n,r.train_win10,r.test_n,r.test_win10,r.test_mean10,r.test_median10))
bench=('weekly_bull','adx20','adx_up','rvol1_5','break20');print('BENCH|'+ '+'.join(bench))
for a,b in [('2018','2020'),('2021','2022'),('2023','2024'),('2025','2026')]:
 q=df.loc[a:b];n,w,m,md=ev(q,bench);print('PERIOD|%s-%s|n=%d|win=%s|mean=%s|median=%s'%(a,b,n,round(float(w),4) if n else None,round(float(m),4) if n else None,round(float(md),4) if n else None))
