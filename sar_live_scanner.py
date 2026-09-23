"""Live CRMS scanner: full-history warmup, PSAR 8H/12H/D/W and daily cross watch."""
import time, requests, numpy as np, pandas as pd
from crms import ASSETS, psar
BASE='https://data-api.binance.vision/api/v3/klines'

def klines(sym, interval, limit=1000):
    r=requests.get(BASE,params={'symbol':sym,'interval':interval,'limit':limit},timeout=20)
    r.raise_for_status(); x=r.json()
    if not x: raise RuntimeError('no rows')
    d=pd.DataFrame(x,columns=['ot','open','high','low','close','volume','ct','q','n','tb','tq','i'])
    for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c])
    d['date']=pd.to_datetime(d.ot,unit='ms',utc=True)
    return d.set_index('date')[['open','high','low','close','volume']]

def state(d):
    p=psar(d); k=len(d)-1
    return bool(p.bull.iloc[k]),float(p.psar.iloc[k]),bool(p.bull.iloc[k-1]) if k else True

def main():
    print('LIVE|START|assets=%d|psar=.02/.20|warmup=1000'%len(ASSETS),flush=True)
    while True:
      now=pd.Timestamp.now(tz='UTC'); print('SCAN|%s'%now.isoformat(),flush=True)
      for sym in ASSETS:
       try:
        # PSAR is recursive: use maximum Binance history window for every timeframe.
        d8=klines(sym,'8h'); d12=klines(sym,'12h'); dd=klines(sym,'1d'); dw=klines(sym,'1w')
        b8,s8,_=state(d8); b12,s12,_=state(d12); bd,sd,prevd=state(dd); bw,sw,_=state(dw)
        px=float(dd.close.iloc[-1]); dist=(px-sd)/px*100 if px else np.nan
        cross=bd and not prevd; breadth=sum([b8,b12,bd,bw])
        tag='CROSS_UP' if cross else ('BULL' if bd else 'BEAR')
        print('LIVE|%s|px=%.8g|8H=%d@%.8g|12H=%d@%.8g|D=%d@%.8g|W=%d@%.8g|breadth=%d/4|distD=%+.2f%%|%s'%(sym,px,b8,s8,b12,s12,bd,sd,bw,sw,breadth,dist,tag),flush=True)
       except Exception as e: print('LIVEFAIL|%s|%s'%(sym,str(e)[:140]),flush=True)
      print('SCAN|DONE',flush=True); time.sleep(3600)

if __name__=='__main__': main()
