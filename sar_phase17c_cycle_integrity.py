"""Phase17C: verify true SAR Daily BUY->SELL cycles.
A trade ends irrevocably at first daily PSAR bearish flip after the bullish entry.
Measure MFE/TOP only inside that SAR cycle and compare with unrestricted 45d path.
"""
import warnings,time,requests,numpy as np,pandas as pd
from crms import ASSETS,fetch,indicators
warnings.filterwarnings('ignore'); BASE='https://data-api.binance.vision/api/v3/klines'; ENTRY_SLIP=.001; EXIT_COST=.002

def hourly(sym,start_ms,end_ms):
 rows=[];ms=start_ms
 while ms<end_ms:
  r=requests.get(BASE,params={'symbol':sym,'interval':'1h','startTime':ms,'endTime':end_ms-1,'limit':1000},timeout=30);r.raise_for_status();x=r.json()
  if not x:break
  rows+=x;ms=x[-1][6]+1
  if len(x)<1000:break
  time.sleep(.02)
 if not rows:return pd.DataFrame()
 d=pd.DataFrame(rows,columns=['ot','open','high','low','close','volume','ct','q','n','tb','tq','i'])
 for c in ['open','high','low','close','volume']:d[c]=pd.to_numeric(d[c])
 d['date']=pd.to_datetime(d.ot,unit='ms',utc=True);return d.set_index('date')[['open','high','low','close','volume']]

EV=[]
for sym in ASSETS:
 try:
  raw=fetch(sym);z=indicators(raw);bull=z.psar_bull.astype(bool);cross=bull&~bull.shift(1,fill_value=False);inds=np.flatnonzero(cross)
  hs=int(raw.index[0].timestamp()*1000);he=int((raw.index[-1]+pd.Timedelta(days=47)).timestamp()*1000);h=hourly(sym,hs,he);print('H1C|%s|bars=%d'%(sym,len(h)),flush=True)
  for i in inds:
   if i<210 or i+1>=len(z):continue
   trigger=float(z.psar.iloc[i-1]);pre=float(z.close.iloc[i-1])
   if not np.isfinite(trigger) or trigger<=0 or bool(bull.iloc[i-1]):continue
   day=z.index[i].floor('D');hh=h[(h.index>=day)&(h.index<day+pd.Timedelta(days=1))];hit=hh[hh.high>=trigger]
   if hit.empty:continue
   ts=hit.index[0];entry=max(trigger,float(hit.iloc[0].open))*(1+ENTRY_SLIP)
   # First DAILY bearish PSAR state after entry. This is the system's irrevocable SELL boundary.
   j=i+1
   while j<len(z) and bool(bull.iloc[j]):j+=1
   sell_day=z.index[j].floor('D') if j<len(z) else None
   # conservative execution: first hourly close on sell day; cycle path ends there
   full=h[(h.index>=ts)&(h.index<ts+pd.Timedelta(hours=1081))]
   if full.empty:continue
   if sell_day is not None:
    sx=full[full.index>=sell_day]
    sell_ts=sx.index[0] if len(sx) else None
   else:sell_ts=None
   cyc=full[full.index<=sell_ts] if sell_ts is not None else full
   if cyc.empty:continue
   topj=int(np.argmax(cyc.high.values));top_ts=cyc.index[topj];mfe=float(cyc.high.iloc[topj]/entry-1)
   ftopj=int(np.argmax(full.high.values));fmfe=float(full.high.iloc[ftopj]/entry-1);ftop_ts=full.index[ftopj]
   if sell_ts is not None:
    sellpx=float(full.loc[sell_ts].close);ret=sellpx/entry-1-EXIT_COST;dur=(sell_ts-ts).total_seconds()/86400.;lag=(sell_ts-top_ts).total_seconds()/86400.
   else:sellpx=ret=dur=lag=np.nan
   EV.append({'sym':sym,'ts':ts,'entry':entry,'sell':sell_ts,'dur':dur,'mfe':mfe,'topday':(top_ts-ts).total_seconds()/86400.,'lag':lag,'ret':ret,'fmfe':fmfe,'ftopday':(ftop_ts-ts).total_seconds()/86400.})
 except Exception as e:print('FAIL17C|%s|%s'%(sym,str(e)[:160]),flush=True)

EV=sorted(EV,key=lambda x:x['ts']);cut=int(len(EV)*.20);O=EV[cut:]
print('EVENTS17C|n=%d|oos=%d'%(len(EV),len(O)),flush=True)
sig=[x for x in O if x['sell'] is not None]
print('SARSELL17C|rate=%.3f|n=%d|dur_med=%.2f|dur_avg=%.2f|top_day_med=%.2f|top_to_sell_med=%.2f|ret_mean=%.4f|ret_med=%.4f|win=%.3f'%(
 len(sig)/len(O),len(sig),np.nanmedian([x['dur'] for x in sig]),np.nanmean([x['dur'] for x in sig]),np.nanmedian([x['topday'] for x in sig]),np.nanmedian([x['lag'] for x in sig]),np.nanmean([x['ret'] for x in sig]),np.nanmedian([x['ret'] for x in sig]),np.mean([x['ret']>0 for x in sig])),flush=True)
for th in [.10,.20,.30,.50]:
 old=[x for x in O if x['fmfe']>=th];valid=[x for x in old if x['mfe']>=th];lost=[x for x in old if x['mfe']<th]
 print('INTEGRITY17C|mfe>=%.2f|unrestricted=%d|within_cycle=%d|valid=%.3f|post_sell_only=%d|postfrac=%.3f'%(th,len(old),len(valid),len(valid)/len(old) if old else np.nan,len(lost),len(lost)/len(old) if old else np.nan),flush=True)
 if valid:
  q=[x for x in valid if x['sell'] is not None]
  print('CYCLE17C|mfe>=%.2f|n=%d|sellrate=%.3f|top_day_med=%.2f|sell_day_med=%.2f|top_to_sell_med=%.2f|ret_mean=%.4f|ret_med=%.4f|win=%.3f'%(
   th,len(valid),len(q)/len(valid),np.nanmedian([x['topday'] for x in valid]),np.nanmedian([x['dur'] for x in q]) if q else np.nan,np.nanmedian([x['lag'] for x in q]) if q else np.nan,np.nanmean([x['ret'] for x in q]) if q else np.nan,np.nanmedian([x['ret'] for x in q]) if q else np.nan,np.mean([x['ret']>0 for x in q]) if q else np.nan),flush=True)
print('DONE17C',flush=True)
