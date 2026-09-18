from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
BASES=['https://api.binance.us/api/v3/klines','https://api.binance.com/api/v3/klines']
ASSETS=['BTCUSDT','ETHUSDT','SOLUSDT','DOTUSDT','AVAXUSDT','SUIUSDT','XRPUSDT','BCHUSDT','LTCUSDT','ICPUSDT','XLMUSDT','ZECUSDT','DYDXUSDT','INJUSDT','NEARUSDT','FILUSDT','LINKUSDT','ADAUSDT','ATOMUSDT','UNIUSDT']
DATA=Path('data');OUT=Path('output')
def fetch(symbol,start='2017-01-01'):
 DATA.mkdir(exist_ok=True);err=None
 for base in BASES:
  try:
   ms=int(pd.Timestamp(start,tz='UTC').timestamp()*1000);rows=[]
   while True:
    r=requests.get(base,params={'symbol':symbol,'interval':'1d','startTime':ms,'limit':1000},timeout=30);r.raise_for_status();x=r.json()
    if not x:break
    rows+=x;ms=x[-1][6]+1
    if len(x)<1000:break
    time.sleep(.05)
   if not rows:raise RuntimeError('no rows')
   cols=['ot','open','high','low','close','volume','ct','q','n','tb','tq','i'];d=pd.DataFrame(rows,columns=cols)
   for c in ['open','high','low','close','volume']:d[c]=pd.to_numeric(d[c])
   d['date']=pd.to_datetime(d.ot,unit='ms',utc=True);d=d.set_index('date');d=d[d.ct<int(pd.Timestamp.now(tz='UTC').timestamp()*1000)]
   print(f'DATA {symbol}: {base} ({len(d)} bars)');return d[['open','high','low','close','volume']]
  except Exception as e:err=e
 raise RuntimeError(err)
def psar(d,step=.02,max_step=.2):
 h,l,c=d.high.values,d.low.values,d.close.values;n=len(d);s=np.full(n,np.nan);b=np.ones(n,dtype=bool)
 if n<3:return pd.DataFrame({'psar':s,'bull':b},index=d.index)
 b[1]=c[1]>=c[0];s[1]=l[0] if b[1] else h[0];ep=h[1] if b[1] else l[1];af=step
 for i in range(2,n):
  q=s[i-1]+af*(ep-s[i-1]);u=b[i-1]
  if u:
   q=min(q,l[i-1],l[i-2])
   if l[i]<q:u=False;q=ep;ep=l[i];af=step
   elif h[i]>ep:ep=h[i];af=min(max_step,af+step)
  else:
   q=max(q,h[i-1],h[i-2])
   if h[i]>q:u=True;q=ep;ep=h[i];af=step
   elif l[i]<ep:ep=l[i];af=min(max_step,af+step)
  s[i]=q;b[i]=u
 return pd.DataFrame({'psar':s,'bull':b},index=d.index)
def indicators(d):
 z=d.copy();p=psar(z);z['psar']=p.psar;z['psar_bull']=p.bull;e12=z.close.ewm(span=12,adjust=False).mean();e26=z.close.ewm(span=26,adjust=False).mean();z['macd']=e12-e26;sig=z.macd.ewm(span=9,adjust=False).mean();z['macd_hist']=z.macd-sig;z['macd_up']=z.macd_hist>z.macd_hist.shift()
 q=z.close.diff();up=q.clip(lower=0);dn=-q.clip(upper=0);rs=up.ewm(alpha=1/14,adjust=False).mean()/dn.ewm(alpha=1/14,adjust=False).mean();z['rsi']=100-100/(1+rs);pc=z.close.shift();tr=pd.concat([z.high-z.low,(z.high-pc).abs(),(z.low-pc).abs()],axis=1).max(axis=1);z['atr']=tr.ewm(alpha=1/14,adjust=False).mean();plus=z.high.diff();minus=-z.low.diff();pdm=plus.where((plus>minus)&(plus>0),0);mdm=minus.where((minus>plus)&(minus>0),0);z['plus_di']=100*pdm.ewm(alpha=1/14,adjust=False).mean()/z.atr;z['minus_di']=100*mdm.ewm(alpha=1/14,adjust=False).mean()/z.atr;dx=100*(z.plus_di-z.minus_di).abs()/(z.plus_di+z.minus_di);z['adx']=dx.ewm(alpha=1/14,adjust=False).mean();z['rvol20']=z.volume/z.volume.rolling(20).mean();return z
def weekly(d):
 w=d.resample('W-SUN',label='right',closed='right').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna();p=psar(w);w['weekly_bull']=p.bull;return w[['weekly_bull']].shift(1).reindex(d.index,method='ffill')
def event_rows(sym,z):
 flip=z.psar_bull.ne(z.psar_bull.shift())&z.psar.notna();rows=[]
 for i in np.flatnonzero(flip.values):
  if i+1>=len(z):continue
  side=1 if z.psar_bull.iloc[i] else -1;entry=z.close.iloc[i];sar=z.psar.iloc[i];r={'symbol':sym,'date':str(z.index[i]),'side':'LONG' if side==1 else 'SHORT','entry':entry,'weekly_ok':bool(z.weekly_bull.iloc[i])==bool(side==1) if pd.notna(z.weekly_bull.iloc[i]) else False,'macd_ok':bool(z.macd_up.iloc[i])==bool(side==1),'di_ok':bool(z.plus_di.iloc[i]>z.minus_di.iloc[i])==bool(side==1),'adx20':z.adx.iloc[i]>=20,'rsi':z.rsi.iloc[i],'rvol20':z.rvol20.iloc[i]}
  end=min(i+5,len(z)-1);zone=max(z.atr.iloc[i],abs(entry-sar)*.25);hit=None
  for k in range(i+1,end+1):
   if z.low.iloc[k]<=sar+zone and z.high.iloc[k]>=sar-zone:hit=k;break
  r['retest5']=hit is not None;r['retest_delay']=hit-i if hit else np.nan;r['retest_entry']=z.close.iloc[hit] if hit else np.nan
  for h in [5,10,20,30,60]:
   r[f'immediate_r{h}']=side*(z.close.iloc[i+h]/entry-1) if i+h<len(z) else np.nan
   r[f'retest_r{h}']=side*(z.close.iloc[hit+h]/z.close.iloc[hit]-1) if hit is not None and hit+h<len(z) else np.nan
  rows.append(r)
 return rows
def summarize(ev):
 rows=[];filters={'ALL':pd.Series(True,index=ev.index),'WEEKLY':ev.weekly_ok,'MACD':ev.macd_ok,'DI':ev.di_ok,'ADX20':ev.adx20,'WEEKLY_MACD_DI':ev.weekly_ok&ev.macd_ok&ev.di_ok,'CONF4':ev.weekly_ok&ev.macd_ok&ev.di_ok&ev.adx20}
 for side in ['LONG','SHORT']:
  base=ev.side.eq(side)
  for name,f in filters.items():
   g=ev[base&f]
   for h in [5,10,20,30,60]:
    x=g[f'immediate_r{h}'].dropna();rt=g[g.retest5][f'retest_r{h}'].dropna();rows.append({'side':side,'filter':name,'horizon':h,'n':len(x),'win':(x>0).mean() if len(x) else np.nan,'mean':x.mean(),'median':x.median(),'retest_rate':g.retest5.mean() if len(g) else np.nan,'retest_n':len(rt),'retest_win':(rt>0).mean() if len(rt) else np.nan,'retest_mean':rt.mean()})
 return pd.DataFrame(rows)
def run_all():
 OUT.mkdir(exist_ok=True);allr=[];fails={}
 for s in ASSETS:
  try:d=fetch(s);z=indicators(d).join(weekly(d));allr+=event_rows(s,z)
  except Exception as e:fails[s]=str(e);print('FAIL',s,e)
 ev=pd.DataFrame(allr);ev.to_csv(OUT/'events.csv',index=False);summary=summarize(ev);summary.to_csv(OUT/'confirmation_retest.csv',index=False);print(summary.to_string(index=False));print('FAILURES',fails)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['run-all']);a.parse_args();run_all()
