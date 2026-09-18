from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

BASE='https://api.binance.com/api/v3/klines'
ASSETS=['BTCUSDT','ETHUSDT','SOLUSDT','DOTUSDT','AVAXUSDT','SUIUSDT','XRPUSDT']
DATA=Path('data'); OUT=Path('output')

def fetch(symbol,start='2017-01-01'):
    DATA.mkdir(exist_ok=True)
    start_ms=int(pd.Timestamp(start,tz='UTC').timestamp()*1000); rows=[]
    while True:
        r=requests.get(BASE,params={'symbol':symbol,'interval':'1d','startTime':start_ms,'limit':1000},timeout=30); r.raise_for_status(); x=r.json()
        if not x: break
        rows.extend(x); start_ms=x[-1][6]+1
        if len(x)<1000: break
        time.sleep(.08)
    cols=['open_time','open','high','low','close','volume','close_time','qv','trades','tb','tq','ignore']
    df=pd.DataFrame(rows,columns=cols)
    for c in ['open','high','low','close','volume']: df[c]=pd.to_numeric(df[c])
    df['date']=pd.to_datetime(df.open_time,unit='ms',utc=True); df=df.set_index('date')
    now=int(pd.Timestamp.now(tz='UTC').timestamp()*1000); df=df[df.close_time<now]
    df[['open','high','low','close','volume']].to_csv(DATA/f'{symbol}.csv')
    return df[['open','high','low','close','volume']]

def psar(df,step=.02,max_step=.2):
    h,l,c=df.high.values,df.low.values,df.close.values; n=len(df)
    sar=np.full(n,np.nan); bull=np.ones(n,dtype=bool)
    if n<3:return pd.DataFrame({'psar':sar,'bull':bull},index=df.index)
    bull[1]=c[1]>=c[0]; sar[1]=l[0] if bull[1] else h[0]; ep=h[1] if bull[1] else l[1]; af=step
    for i in range(2,n):
        s=sar[i-1]+af*(ep-sar[i-1]); b=bull[i-1]
        if b:
            s=min(s,l[i-1],l[i-2])
            if l[i]<s: b=False;s=ep;ep=l[i];af=step
            elif h[i]>ep: ep=h[i];af=min(max_step,af+step)
        else:
            s=max(s,h[i-1],h[i-2])
            if h[i]>s: b=True;s=ep;ep=h[i];af=step
            elif l[i]<ep: ep=l[i];af=min(max_step,af+step)
        sar[i]=s;bull[i]=b
    return pd.DataFrame({'psar':sar,'bull':bull},index=df.index)

def indicators(df):
    z=df.copy(); p=psar(z);z['psar']=p.psar;z['psar_bull']=p.bull
    e12=z.close.ewm(span=12,adjust=False).mean();e26=z.close.ewm(span=26,adjust=False).mean();z['macd']=e12-e26;z['macd_signal']=z.macd.ewm(span=9,adjust=False).mean();z['macd_hist']=z.macd-z.macd_signal
    d=z.close.diff();up=d.clip(lower=0);dn=-d.clip(upper=0);rs=up.ewm(alpha=1/14,adjust=False).mean()/dn.ewm(alpha=1/14,adjust=False).mean();z['rsi']=100-100/(1+rs)
    pc=z.close.shift();tr=pd.concat([(z.high-z.low),(z.high-pc).abs(),(z.low-pc).abs()],axis=1).max(axis=1);z['atr']=tr.ewm(alpha=1/14,adjust=False).mean()
    plus=z.high.diff();minus=-z.low.diff();pdm=plus.where((plus>minus)&(plus>0),0);mdm=minus.where((minus>plus)&(minus>0),0);atr=z.atr
    z['plus_di']=100*pdm.ewm(alpha=1/14,adjust=False).mean()/atr;z['minus_di']=100*mdm.ewm(alpha=1/14,adjust=False).mean()/atr;dx=100*(z.plus_di-z.minus_di).abs()/(z.plus_di+z.minus_di);z['adx']=dx.ewm(alpha=1/14,adjust=False).mean()
    for n in [20,50,100,200]: z[f'ema{n}']=z.close.ewm(span=n,adjust=False).mean()
    z['rvol20']=z.volume/z.volume.rolling(20).mean();return z

def weekly_regime(df):
    w=df.resample('W-SUN',label='right',closed='right').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna();p=psar(w);w['weekly_psar']=p.psar;w['weekly_bull']=p.bull
    # shift one completed weekly observation before mapping to daily: prevents using an unfinished current week
    return w[['weekly_psar','weekly_bull']].shift(1).reindex(df.index,method='ffill')

def events(symbol,z):
    flips=z.psar_bull.ne(z.psar_bull.shift()) & z.psar.notna(); out=[]
    for i in np.flatnonzero(flips.values):
        if i+1>=len(z):continue
        side=1 if z.psar_bull.iloc[i] else -1; entry=z.close.iloc[i]
        row={'symbol':symbol,'date':str(z.index[i]),'side':'LONG' if side==1 else 'SHORT','entry':entry,'weekly_bull':bool(z.weekly_bull.iloc[i]) if pd.notna(z.weekly_bull.iloc[i]) else None,'rsi':z.rsi.iloc[i],'adx':z.adx.iloc[i],'macd_hist':z.macd_hist.iloc[i],'rvol20':z.rvol20.iloc[i]}
        for h in [1,3,5,10,20,30,60]: row[f'r{h}']=side*(z.close.iloc[i+h]/entry-1) if i+h<len(z) else np.nan
        j=min(i+60,len(z)-1); hi=z.high.iloc[i+1:j+1].max();lo=z.low.iloc[i+1:j+1].min();row['mfe60']=(hi/entry-1) if side==1 else (entry/lo-1);row['mae60']=(lo/entry-1) if side==1 else (entry/hi-1)
        out.append(row)
    return out

def run_all():
    OUT.mkdir(exist_ok=True); all_events=[]; latest={}
    for s in ASSETS:
        d=fetch(s);z=indicators(d);z=z.join(weekly_regime(d));all_events+=events(s,z);r=z.iloc[-1]
        latest[s]={k:(bool(r[k]) if k in ['psar_bull','weekly_bull'] and pd.notna(r[k]) else (None if pd.isna(r[k]) else float(r[k]))) for k in ['close','psar','psar_bull','weekly_psar','weekly_bull','rsi','adx','plus_di','minus_di','macd_hist','rvol20']}
    ev=pd.DataFrame(all_events);ev.to_csv(OUT/'events.csv',index=False)
    summary=[]
    for (sym,side),g in ev.groupby(['symbol','side']):
        for h in [5,10,20,30,60]:
            x=g[f'r{h}'].dropna();summary.append({'symbol':sym,'side':side,'horizon':h,'n':len(x),'win_rate':(x>0).mean(),'mean':x.mean(),'median':x.median(),'mean_mfe60':g.mfe60.mean(),'mean_mae60':g.mae60.mean()})
    pd.DataFrame(summary).to_csv(OUT/'backtest_summary.csv',index=False)
    (OUT/'latest.json').write_text(json.dumps({'generated_at':pd.Timestamp.now(tz='UTC').isoformat(),'assets':latest},indent=2))
    print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['run-all']);a=ap.parse_args();run_all()
