"""Phase 6: realistic executable walk-forward strategy.
Signal: bullish daily SAR flip filtered by walk-forward HGB.
Execution: next-day OPEN (1-bar lag); TP 10/12%; bearish SAR flip safety exit at next-day OPEN.
Includes round-trip fees+slippage. Compares HGB >=.60 and probability bucket [.65,.70).
"""
import warnings,numpy as np,pandas as pd
from crms import ASSETS,fetch,indicators,weekly
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings('ignore'); rows=[]
COST_RT=0.004  # 40 bps round trip: fees + slippage stress assumption
for s in ASSETS:
 try:
  raw=fetch(s); z=indicators(raw).join(weekly(raw)); c=z.close; v=z.volume
  for n in [3,5,10,20,40,60,84]: z[f'r{n}']=c.pct_change(n)
  for n in [20,50,100,200]: z[f'ema{n}']=c.ewm(span=n,adjust=False).mean()
  z['atrp']=z.atr/c; z['di_gap']=z.plus_di-z.minus_di; z['bbw']=4*c.rolling(20).std()/c.rolling(20).mean(); z['obv']=(np.sign(c.diff()).fillna(0)*v).cumsum(); z['dd50']=c/c.rolling(50).max()-1
  bull=z.psar_bull.astype(bool); en=bull&~bull.shift(1,fill_value=False); ex=~bull&bull.shift(1,fill_value=False); E=np.flatnonzero(en); X=np.flatnonzero(ex); bear=~bull; age=bear.groupby(bear.ne(bear.shift()).cumsum()).cumcount()+1
  for sig in E:
   if sig<210 or sig+1>=len(z): continue
   qq=[j for j in X if j>sig]
   if not qq: continue
   flip=qq[0]; p=sig-1; entry_i=sig+1; entry=float(z.open.iloc[entry_i])
   # safety exit is next open after bearish flip where available, otherwise flip close
   exit_i=min(flip+1,len(z)-1); safety=float(z.open.iloc[exit_i]) if exit_i>flip else float(z.close.iloc[flip])
   def x(col,k=0):
    a=z[col].iloc[p-k]; return float(a) if pd.notna(a) else np.nan
   r={'date':z.index[sig],'symbol':s,'sig_i':sig,'entry_i':entry_i,'flip_i':flip,'exit_i':exit_i,'entry':entry,'safety':safety,'bear_age':float(age.iloc[p]),'weekly':int(bool(z.weekly_bull.iloc[sig])) if pd.notna(z.weekly_bull.iloc[sig]) else 0,'adx':x('adx'),'adx_d3':x('adx')-x('adx',3),'di':x('di_gap'),'di_d3':x('di_gap')-x('di_gap',3),'rsi':x('rsi'),'rsi_d3':x('rsi')-x('rsi',3),'hist':x('macd_hist'),'hist_d3':x('macd_hist')-x('macd_hist',3),'rvol':x('rvol20'),'atrp':x('atrp'),'bbw':x('bbw'),'bbw_d5':x('bbw')-x('bbw',5),'obv_d10':x('obv')-x('obv',10),'dd50':x('dd50')}
   for n in [5,10,20,40,60,84]: r[f'r{n}']=x(f'r{n}')
   for n in [20,50,100,200]: r[f'ema{n}gap']=x('close')/x(f'ema{n}')-1
   # label remains original meta-label: profitable SAR cycle, known only for training history
   r['label']=int(float(z.close.iloc[flip])/float(z.close.iloc[sig])-1>0)
   # realistic executable outcomes. TP triggered from entry day through bearish flip day.
   for target in [.10,.12]:
    tp=entry*(1+target); hit=None
    for k in range(entry_i,flip+1):
     if float(z.high.iloc[k])>=tp: hit=k; break
    if hit is not None:
     gross=target; hold=hit-entry_i+1; reason='TP'
    else:
     gross=safety/entry-1; hold=exit_i-entry_i; reason='SAR'
    r[f'ret{int(target*100)}']=gross-COST_RT; r[f'hold{int(target*100)}']=hold; r[f'reason{int(target*100)}']=reason
   rows.append(r)
 except Exception as e: print('FAIL|%s|%s'%(s,str(e)[:100]),flush=True)
t=pd.DataFrame(rows).set_index('date').sort_index(); btc=indicators(fetch('BTCUSDT')); bc=btc.close; B=pd.DataFrame({'btc20':bc.pct_change(20),'btc60':bc.pct_change(60),'btc84':bc.pct_change(84),'btcrsi':btc.rsi,'btcadx':btc.adx,'btcdd50':bc/bc.rolling(50).max()-1},index=btc.index)
for k in B: t[k]=B[k].reindex(t.index).ffill().to_numpy()
t['rs20']=t.r20-t.btc20; t['rs60']=t.r60-t.btc60
exclude=['symbol','sig_i','entry_i','flip_i','exit_i','entry','safety','label','ret10','ret12','hold10','hold12','reason10','reason12']; features=[c for c in t.columns if c not in exclude]; X=t[features].astype(float); y=t.label.to_numpy(); idx=pd.DatetimeIndex(pd.to_datetime(t.index,utc=True)).tz_convert(None); D=idx.unique().sort_values(); pred=np.full(len(t),np.nan); bounds=[.50,.60,.70,.80,.90,1.0]
for f in range(5):
 train_end=D[int(len(D)*bounds[f])]; test_end=D[-1]+pd.Timedelta(days=1) if f==4 else D[int(len(D)*bounds[f+1])]; tr=idx<train_end; te=(idx>=train_end)&(idx<test_end)
 m=make_pipeline(SimpleImputer(),HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,learning_rate=.04,l2_regularization=3,min_samples_leaf=25,random_state=7)); m.fit(X.loc[tr],y[tr]); pred[te]=m.predict_proba(X.loc[te])[:,1]; print('FOLD6|%d|train=%d|test=%d'%(f+1,tr.sum(),te.sum()),flush=True)
t['prob']=pred
for name,q in [('GE60',t.prob>=.60),('B65_70',(t.prob>=.65)&(t.prob<.70))]:
 a=t[q]
 for target in [10,12]:
  rr=a[f'ret{target}']; tp=a[f'reason{target}'].eq('TP');
  print('STRAT|%s|tp=%d|n=%d|win=%.3f|tp_rate=%.3f|mean=%.4f|median=%.4f|hold_med=%.1f|hold_mean=%.1f|sumret=%.4f'%(name,target,len(a),(rr>0).mean(),tp.mean(),rr.mean(),rr.median(),a[f'hold{target}'].median(),a[f'hold{target}'].mean(),rr.sum()),flush=True)
  # chronological compounding, one-unit sequential-return diagnostic (not portfolio overlap adjusted)
  print('COMPOUND|%s|tp=%d|factor=%.3f|maxloss=%.3f'%(name,target,float(np.prod(1+rr.dropna())),rr.min()),flush=True)
 # per fold stability for preferred 12 target
 for f in range(5):
  lo=D[int(len(D)*bounds[f])]; hi=D[-1]+pd.Timedelta(days=1) if f==4 else D[int(len(D)*bounds[f+1])]; g=a[(pd.DatetimeIndex(pd.to_datetime(a.index,utc=True)).tz_convert(None)>=lo)&(pd.DatetimeIndex(pd.to_datetime(a.index,utc=True)).tz_convert(None)<hi)]; rr=g.ret12
  print('STABLE|%s|fold=%d|n=%d|win12=%.3f|mean12=%.4f'%(name,f+1,len(g),(rr>0).mean() if len(g) else np.nan,rr.mean() if len(g) else np.nan),flush=True)
print('COST_RT|%.4f'%COST_RT,flush=True); print('DONE6',flush=True)
