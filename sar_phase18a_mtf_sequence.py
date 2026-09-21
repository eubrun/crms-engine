"""Phase18A: multi-timeframe PSAR sequence around true SAR-cycle TOP.
Build true Daily bullish cycles; examine PSAR 1H/2H/4H/6H/8H/12H from TOP-3d to TOP+3d.
Search causal bearish-state combinations and persistence that anticipate Daily SELL while controlling premature signals.
"""
import warnings,time,requests,itertools,numpy as np,pandas as pd
from crms import ASSETS,fetch,indicators,psar
warnings.filterwarnings('ignore');BASE='https://data-api.binance.vision/api/v3/klines';TFS=[1,2,4,6,8,12]
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
def mtf(h):
 out=pd.DataFrame(index=h.index)
 for k in TFS:
  if k==1:r=h
  else:r=h.resample('%dh'%k,label='right',closed='right',origin='start_day').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
  p=psar(r);b=pd.Series(p.bull,index=r.index)
  # only completed higher-TF bars are propagated; no look-ahead
  out['b%d'%k]=b.reindex(h.index,method='ffill').astype(bool)
 return out
C=[]
for sym in ASSETS:
 try:
  d=fetch(sym);z=indicators(d);bull=z.psar_bull.astype(bool);cross=bull&~bull.shift(1,fill_value=False)
  h=hourly(sym,int(d.index[0].timestamp()*1000),int((d.index[-1]+pd.Timedelta(days=2)).timestamp()*1000));m=mtf(h);print('H18A|%s|%d'%(sym,len(h)),flush=True)
  for i in np.flatnonzero(cross.values):
   if i<210:continue
   j=i+1
   while j<len(z) and bool(bull.iloc[j]):j+=1
   if j>=len(z):continue
   buyday=z.index[i].floor('D');sellday=z.index[j].floor('D');hh=h[(h.index>=buyday)&(h.index<=sellday)]
   if len(hh)<24:continue
   top=hh.high.idxmax();entry=float(z.close.iloc[i]);mfe=float(hh.high.max()/entry-1)
   # first H1 observation on Daily sell day = benchmark sell time
   sx=hh[hh.index>=sellday];sellts=sx.index[0] if len(sx) else hh.index[-1]
   C.append({'sym':sym,'buy':hh.index[0],'sell':sellts,'top':top,'mfe':mfe,'h':hh,'m':m.loc[hh.index]})
 except Exception as e:print('FAIL18A|%s|%s'%(sym,str(e)[:140]),flush=True)
C=sorted(C,key=lambda x:x['buy']);O=C[int(.2*len(C)):]
print('EVENTS18A|n=%d|oos=%d'%(len(C),len(O)),flush=True)
# descriptive flip timing around top
for th in [.10,.20,.30,.50]:
 g=[c for c in O if c['mfe']>=th];print('BUCKET18A|%.2f|n=%d'%(th,len(g)),flush=True)
 for k in TFS:
  vals=[]
  for c in g:
   s=c['m']['b%d'%k];w=s[(s.index>=c['top']-pd.Timedelta(days=3))&(s.index<=min(c['sell'],c['top']+pd.Timedelta(days=3)))]
   # first bullish->bearish flip in window
   flip=(~w)&w.shift(1,fill_value=True);ix=w.index[flip]
   if len(ix):vals.append((ix[0]-c['top']).total_seconds()/86400.)
  print('FLIP18A|mfe%.2f|tf%dh|found%.3f|rel_top_med%.2f'%(th,k,len(vals)/len(g) if g else np.nan,np.median(vals) if vals else np.nan),flush=True)
# causal rules: N simultaneous bearish TFs, optionally persistence 6/12/24h; first occurrence after trade has +10% MFE-to-date
rules=[]
for n in range(2,7):
 for persist in [1,6,12,24]:rules.append((n,persist))
for th in [.10,.20,.30,.50]:
 g=[c for c in O if c['mfe']>=th]
 for n,persist in rules:
  rec=[]
  for c in g:
   mm=c['m'];hh=c['h'];bear=(~mm[['b%d'%k for k in TFS]]).sum(axis=1)>=n
   # activation only after running high has reached +10%, causal
   armed=(hh.high.cummax()/float(hh.open.iloc[0])-1)>=.10
   q=(bear&armed).rolling(persist,min_periods=persist).sum()>=persist;ix=q.index[q]
   if not len(ix):continue
   t=ix[0];rel=(t-c['top']).total_seconds()/86400.;lead=(c['sell']-t).total_seconds()/86400.;prem=t<c['top']
   # false/premature if signal occurs >24h before eventual top or price later makes >3% higher high
   later=hh[hh.index>t];higher=(float(later.high.max()/hh.loc[:t].high.max()-1) if len(later) else 0)
   false=(rel < -1.0) or (higher>.03)
   rec.append((rel,lead,prem,false,higher))
  if rec:
   a=np.array(rec,float);print('SEQ18A|mfe%.2f|bearN%d|persist%d|signal%.3f|relTopMed%.2f|leadSellMed%.2f|preTop%.3f|false%.3f|laterHighMed%.3f'%(th,n,persist,len(rec)/len(g) if g else np.nan,np.median(a[:,0]),np.median(a[:,1]),np.mean(a[:,2]),np.mean(a[:,3]),np.median(a[:,4])),flush=True)
print('DONE18A',flush=True)
