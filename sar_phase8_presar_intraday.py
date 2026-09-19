"""Phase 8: PRE-SAR -> intraday cross -> protected exit.
Daily data define the *known before trigger* daily SAR level and pre-cross approach features.
1h Binance public mirror reconstructs the first intraday bar whose HIGH crosses that fixed daily SAR.
Entry is conservatively max(SAR trigger, hourly open) + entry slippage; no end-of-day information is used.
Walk-forward HGB evaluates targetability; execution grid tests TP/SL/time exits.
"""
import warnings,time,requests,numpy as np,pandas as pd
from crms import ASSETS,fetch,indicators,weekly
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings('ignore'); BASE='https://data-api.binance.vision/api/v3/klines'; COST_EXIT=.002; ENTRY_SLIP=.001

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

def slope(a):
 y=np.asarray(a,float);x=np.arange(len(y));return float(np.polyfit(x,y,1)[0]) if np.isfinite(y).all() else np.nan
rows=[]
for sym in ASSETS:
 try:
  raw=fetch(sym);z=indicators(raw).join(weekly(raw));c=z.close;v=z.volume;bull=z.psar_bull.astype(bool);cross=bull&~bull.shift(1,fill_value=False);inds=np.flatnonzero(cross)
  # one hourly pull for full asset history, then slice cross windows
  hs=int(raw.index[0].timestamp()*1000);he=int((raw.index[-1]+pd.Timedelta(days=2)).timestamp()*1000);h=hourly(sym,hs,he);print('H1|%s|bars=%d'%(sym,len(h)),flush=True)
  for i in inds:
   if i<210 or i+16>=len(z):continue
   # SAR trigger must be knowable from pre-day state. Use bullish flip threshold represented by prior bearish SAR.
   trigger=float(z.psar.iloc[i-1]); preclose=float(c.iloc[i-1]);
   if not np.isfinite(trigger) or trigger<=0:continue
   day=z.index[i].floor('D');hh=h[(h.index>=day)&(h.index<day+pd.Timedelta(days=1))]
   hit=hh[hh.high>=trigger]
   if hit.empty:continue
   ts=hit.index[0];bar=hit.iloc[0];entry=max(trigger,float(bar.open))*(1+ENTRY_SLIP)
   # pre-SAR distance: positive means SAR above close
   dist=[]
   for k in range(7,0,-1):dist.append((float(z.psar.iloc[i-k])-float(c.iloc[i-k]))/float(c.iloc[i-k]))
   # require bearish immediately pre-cross; retain all valid historical flips otherwise
   if bool(bull.iloc[i-1]):continue
   r={'date':z.index[i],'symbol':sym,'i':i,'trigger':trigger,'entry':entry,'cross_hour':ts.hour,'gap_to_sar':trigger/preclose-1,'d1':dist[-1],'d3':dist[-3],'d5':dist[-5],'d7':dist[-7],'approach3':dist[-3]-dist[-1],'approach5':dist[-5]-dist[-1],'approach7':dist[-7]-dist[-1],'dist_slope3':slope(dist[-3:]),'dist_slope5':slope(dist[-5:]),'dist_slope7':slope(dist),'conv3':int(all(np.diff(dist[-3:])<0)),'conv5':int(all(np.diff(dist[-5:])<0)),'sar_slope3':slope(z.psar.iloc[i-3:i].values/c.iloc[i-3:i].values),'ret3':float(c.iloc[i-1]/c.iloc[i-4]-1),'ret5':float(c.iloc[i-1]/c.iloc[i-6]-1),'ret10':float(c.iloc[i-1]/c.iloc[i-11]-1),'rsi':float(z.rsi.iloc[i-1]),'adx':float(z.adx.iloc[i-1]),'di_gap':float(z.plus_di.iloc[i-1]-z.minus_di.iloc[i-1]),'hist':float(z.macd_hist.iloc[i-1]),'hist_d3':float(z.macd_hist.iloc[i-1]-z.macd_hist.iloc[i-4]),'rvol':float(z.rvol20.iloc[i-1]),'atrp':float(z.atr.iloc[i-1]/c.iloc[i-1]),'weekly':int(bool(z.weekly_bull.iloc[i-1])) if pd.notna(z.weekly_bull.iloc[i-1]) else 0}
   # future hourly path from trigger time up to 15 days; target label +10% before -6% or 10d time close
   path=h[(h.index>=ts)&(h.index<day+pd.Timedelta(days=16))]
   r['_path']=path
   # model label: +10% reached before -6%, conservative same-hour ambiguity = failure
   up=entry*1.10;dn=entry*.94;lab=0
   for _,b in path[path.index<day+pd.Timedelta(days=11)].iterrows():
    hu=b.high>=up;hd=b.low<=dn
    if hd:lab=0;break
    if hu:lab=1;break
   r['label']=lab;rows.append(r)
 except Exception as e:print('FAIL8|%s|%s'%(sym,str(e)[:100]),flush=True)
t=pd.DataFrame(rows).set_index('date').sort_index();print('EVENTS8|n=%d|assets=%d|labelrate=%.3f'%(len(t),t.symbol.nunique(),t.label.mean()),flush=True)
# BTC regime features known at prior daily close
btc=indicators(fetch('BTCUSDT'));bc=btc.close;B=pd.DataFrame({'btc20':bc.pct_change(20),'btc60':bc.pct_change(60),'btcrsi':btc.rsi,'btcadx':btc.adx,'btc_above50':(bc>bc.ewm(span=50,adjust=False).mean()).astype(int),'btc_sarbull':btc.psar_bull.astype(int)},index=btc.index).shift(1)
for k in B:t[k]=B[k].reindex(t.index).ffill().to_numpy()
exclude=['symbol','i','trigger','entry','_path','label'];features=[c for c in t.columns if c not in exclude];X=t[features].astype(float);y=t.label.to_numpy();idx=pd.DatetimeIndex(pd.to_datetime(t.index,utc=True)).tz_convert(None);D=idx.unique().sort_values();pred=np.full(len(t),np.nan);bounds=[.50,.60,.70,.80,.90,1.0]
for f in range(5):
 tr_end=D[int(len(D)*bounds[f])];te_end=D[-1]+pd.Timedelta(days=1) if f==4 else D[int(len(D)*bounds[f+1])];tr=idx<tr_end;te=(idx>=tr_end)&(idx<te_end);m=make_pipeline(SimpleImputer(),HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,learning_rate=.04,l2_regularization=4,min_samples_leaf=25,random_state=8));m.fit(X.loc[tr],y[tr]);pred[te]=m.predict_proba(X.loc[te])[:,1];print('FOLD8|%d|train=%d|test=%d'%(f+1,tr.sum(),te.sum()),flush=True)
t['prob']=pred

def sim(r,tp,sl,H):
 e=float(r.entry);up=e*(1+tp);dn=e*(1-sl);p=r['_path'];end=pd.Timestamp(r.name)+pd.Timedelta(days=H+1);p=p[p.index<end]
 for _,b in p.iterrows():
  hu=b.high>=up;hd=b.low<=dn
  if hu and hd:return -sl-ENTRY_SLIP-COST_EXIT,'SLAMB'
  if hd:return -sl-ENTRY_SLIP-COST_EXIT,'SL'
  if hu:return tp-ENTRY_SLIP-COST_EXIT,'TP'
 if len(p):return float(p.close.iloc[-1]/e-1)-ENTRY_SLIP-COST_EXIT,'TIME'
 return np.nan,'NONE'
configs=[(tp,sl,H) for tp in [.08,.10,.12,.15] for sl in [.04,.05,.06,.075,.10] for H in [3,5,7,10,12,15]]
for th in [.45,.50,.55,.60]:
 a=t[t.prob>=th];print('COUNT8|th=%.2f|n=%d'%(th,len(a)),flush=True);scores=[]
 for cfg in configs:
  out=[sim(r,*cfg) for _,r in a.iterrows()];rr=np.array([q[0] for q in out if np.isfinite(q[0])]);score=rr.mean()+.03*(rr>0).mean()+.15*rr.min() if len(rr) else -9;scores.append((score,cfg,rr))
 for rank,(score,cfg,rr) in enumerate(sorted(scores,key=lambda q:q[0],reverse=True)[:8],1):print('TOP8|th=%.2f|rank=%d|tp=%.3f|sl=%.3f|H=%d|n=%d|win=%.3f|mean=%.4f|med=%.4f|worst=%.4f|score=%.4f'%((th,rank)+cfg+(len(rr),(rr>0).mean(),rr.mean(),np.median(rr),rr.min(),score)),flush=True)
# feature diagnostic via permutation-free HGB gain unavailable; report approach quartiles OOS relationship
q=t[np.isfinite(t.prob)].copy();q['aq']=pd.qcut(q.approach5,4,duplicates='drop')
for b,g in q.groupby('aq',observed=True):print('APPROACH8|%s|n=%d|label=%.3f|pmean=%.3f'%(str(b),len(g),g.label.mean(),g.prob.mean()),flush=True)
print('DONE8',flush=True)
