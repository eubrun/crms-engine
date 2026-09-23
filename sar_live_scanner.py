"""Live CRMS scanner: closed-candle PSAR + separate live price cross."""
import time, requests, numpy as np, pandas as pd
from crms import ASSETS, psar
BASE='https://data-api.binance.vision/api/v3/klines'

def klines(sym, interval, limit=1000):
    r=requests.get(BASE,params={'symbol':sym,'interval':interval,'limit':limit},timeout=20); r.raise_for_status(); x=r.json()
    if not x: raise RuntimeError('no rows')
    cols=['ot','open','high','low','close','volume','ct','q','n','tb','tq','i']
    d=pd.DataFrame(x,columns=cols)
    for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c])
    d['date']=pd.to_datetime(d.ot,unit='ms',utc=True); d['ct']=pd.to_numeric(d.ct)
    return d.set_index('date')

def closed(d):
    now=int(pd.Timestamp.now(tz='UTC').timestamp()*1000)
    z=d[d.ct < now][['open','high','low','close','volume']]
    if len(z)<3: raise RuntimeError('insufficient closed bars')
    return z

def closed_state(raw):
    d=closed(raw); p=psar(d); k=len(d)-1
    return bool(p.bull.iloc[k]),float(p.psar.iloc[k]),bool(p.bull.iloc[k-1]),d

def live_price(raw): return float(raw.close.iloc[-1])

def main():
    print('LIVE|START|assets=%d|psar=.02/.20|CLOSED_CANDLE_STATE=1'%len(ASSETS),flush=True)
    while True:
      print('SCAN|%s'%pd.Timestamp.now(tz='UTC').isoformat(),flush=True)
      for sym in ASSETS:
       try:
        r8=klines(sym,'8h'); r12=klines(sym,'12h'); rd=klines(sym,'1d'); rw=klines(sym,'1w')
        b8,s8,_,_=closed_state(r8); b12,s12,_,_=closed_state(r12); bd,sd,prevd,_=closed_state(rd); bw,sw,prevw,_=closed_state(rw)
        px=live_price(rd)
        # Confirmed state = last CLOSED candle, matching backtest. Live cross is only an alert against the frozen closed-candle SAR level.
        liveD=(px>sd) if not bd else (px>=sd)
        liveW=(px>sw) if not bw else (px>=sw)
        dist=(px-sd)/px*100 if px else np.nan
        cross_conf=bd and not prevd
        tag='CROSS_CONF' if cross_conf else ('BULL' if bd else 'BEAR')
        print('LIVE|%s|px=%.8g|8Hc=%d@%.8g|12Hc=%d@%.8g|Dc=%d@%.8g|Wc=%d@%.8g|Dlive=%d|Wlive=%d|distD=%+.2f%%|%s'%(sym,px,b8,s8,b12,s12,bd,sd,bw,sw,liveD,liveW,dist,tag),flush=True)
       except Exception as e: print('LIVEFAIL|%s|%s'%(sym,str(e)[:160]),flush=True)
      print('SCAN|DONE',flush=True); time.sleep(3600)

if __name__=='__main__': main()
