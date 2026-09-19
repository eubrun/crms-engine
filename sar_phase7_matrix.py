"""Phase 7: broaden opportunities and optimize protection without using bearish SAR as sole exit.
Walk-forward HGB; entry modes signal-day close and next-day open. Intraday SAR-cross entry is not claimed from daily OHLC because exact crossing price/path is not reconstructable reliably. Grid TP/SL/time-exit; conservative same-bar TP+SL resolution = SL. OOS only.
"""
import warnings,numpy as np,pandas as pd
from crms import ASSETS,fetch,indicators,weekly
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings('ignore');R=[];COST=.004
for s in ASSETS:
 try:
  z=indicators(fetch(s)).join(weekly(fetch(s)));c=z.close;v=z.volume
  for n in [3,5,10,20,40,60,84]:z[f'r{n}']=c.pct_change(n)
  for n in [20,50,100,200]:z[f'ema{n}']=c.ewm(span=n,adjust=False).mean()
  z['atrp']=z.atr/c;z['di_gap']=z.plus_di-z.minus_di;z['bbw']=4*c.rolling(20).std()/c.rolling(20).mean();z['obv']=(np.sign(c.diff()).fillna(0)*v).cumsum();z['dd50']=c/c.rolling(50).max()-1
  bull=z.psar_bull.astype(bool);en=bull&~bull.shift(1,fill_value=False);ex=~bull&bull.shift(1,fill_value=False);E=np.flatnonzero(en);X=np.flatnonzero(ex);bear=~bull;age=bear.groupby(bear.ne(bear.shift()).cumsum()).cumcount()+1
  for sig in E:
   if sig<210 or sig+1>=len(z):continue
   qq=[j for j in X if j>sig]
   if not qq:continue
   flip=qq[0];p=sig-1
   def x(col,k=0):
    a=z[col].iloc[p-k];return float(a) if pd.notna(a) else np.nan
   r={'date':z.index[sig],'symbol':s,'sig':sig,'flip':flip,'close_entry':float(z.close.iloc[sig]),'nextopen_entry':float(z.open.iloc[sig+1]),'bear_age':float(age.iloc[p]),'weekly':int(bool(z.weekly_bull.iloc[sig])) if pd.notna(z.weekly_bull.iloc[sig]) else 0,'adx':x('adx'),'adx_d3':x('adx')-x('adx',3),'di':x('di_gap'),'di_d3':x('di_gap')-x('di_gap',3),'rsi':x('rsi'),'rsi_d3':x('rsi')-x('rsi',3),'hist':x('macd_hist'),'hist_d3':x('macd_hist')-x('macd_hist',3),'rvol':x('rvol20'),'atrp':x('atrp'),'bbw':x('bbw'),'bbw_d5':x('bbw')-x('bbw',5),'obv_d10':x('obv')-x('obv',10),'dd50':x('dd50'),'label':int(float(z.close.iloc[flip])/float(z.close.iloc[sig])-1>0)}
   for n in [5,10,20,40,60,84]:r[f'r{n}']=x(f'r{n}')
   for n in [20,50,100,200]:r[f'ema{n}gap']=x('close')/x(f'ema{n}')-1
   # save path arrays needed for execution simulation
   r['_z']=z;R.append(r)
 except Exception as e:print('FAIL7|%s|%s'%(s,str(e)[:80]),flush=True)
t=pd.DataFrame(R).set_index('date').sort_index();btc=indicators(fetch('BTCUSDT'));bc=btc.close;B=pd.DataFrame({'btc20':bc.pct_change(20),'btc60':bc.pct_change(60),'btc84':bc.pct_change(84),'btcrsi':btc.rsi,'btcadx':btc.adx,'btcdd50':bc/bc.rolling(50).max()-1},index=btc.index)
for k in B:t[k]=B[k].reindex(t.index).ffill().to_numpy()
t['rs20']=t.r20-t.btc20;t['rs60']=t.r60-t.btc60
exclude=['symbol','sig','flip','close_entry','nextopen_entry','label','_z'];features=[c for c in t.columns if c not in exclude];X=t[features].astype(float);y=t.label.to_numpy();idx=pd.DatetimeIndex(pd.to_datetime(t.index,utc=True)).tz_convert(None);D=idx.unique().sort_values();pred=np.full(len(t),np.nan);bounds=[.50,.60,.70,.80,.90,1.0]
for f in range(5):
 te0=D[int(len(D)*bounds[f])];te1=D[-1]+pd.Timedelta(days=1) if f==4 else D[int(len(D)*bounds[f+1])];tr=idx<te0;te=(idx>=te0)&(idx<te1);m=make_pipeline(SimpleImputer(),HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,learning_rate=.04,l2_regularization=3,min_samples_leaf=25,random_state=7));m.fit(X.loc[tr],y[tr]);pred[te]=m.predict_proba(X.loc[te])[:,1];print('FOLD7|%d|train=%d|test=%d'%(f+1,tr.sum(),te.sum()),flush=True)
t['prob']=pred
# simulate deterministic exit grid; close entry starts next bar for TP/SL to avoid using same daily bar after close
configs=[]
for mode in ['CLOSE0','OPEN1']:
 for tp in [.08,.10,.12,.15]:
  for sl in [.04,.05,.06,.075,.10]:
   for horizon in [5,7,10,12,15]:configs.append((mode,tp,sl,horizon))
def sim(row,mode,tp,sl,H):
 z=row['_z'];sig=int(row.sig);flip=int(row.flip);entry=float(row.close_entry if mode=='CLOSE0' else row.nextopen_entry);start=sig+1;last=min(start+H-1,flip,len(z)-1);up=entry*(1+tp);dn=entry*(1-sl)
 for k in range(start,last+1):
  hitu=float(z.high.iloc[k])>=up;hitd=float(z.low.iloc[k])<=dn
  if hitu and hitd:return -sl-COST,k-start+1,'SLAMB'
  if hitd:return -sl-COST,k-start+1,'SL'
  if hitu:return tp-COST,k-start+1,'TP'
 # time exit at close, or earlier bearish flip close
 return float(z.close.iloc[last]/entry-1)-COST,last-start+1,'TIME/SAR'
for threshold in [.50,.55,.60]:
 a=t[t.prob>=threshold]
 print('COUNT7|th=%.2f|n=%d'%(threshold,len(a)),flush=True)
 scored=[]
 for cfg in configs:
  vals=[sim(r,*cfg) for _,r in a.iterrows()];rr=np.array([v[0] for v in vals]);holds=np.array([v[1] for v in vals]);wins=(rr>0).mean();mean=rr.mean();med=np.median(rr);worst=rr.min();exp=mean
  # score rewards expectancy and win rate, penalizes tail loss; no arbitrary max-WR selection
  score=exp+0.03*wins+0.15*worst
  scored.append((score,cfg,len(rr),wins,mean,med,worst,np.median(holds)))
 for rank,q in enumerate(sorted(scored,reverse=True)[:10],1):
  score,cfg,n,wr,mean,med,worst,hm=q;print('TOP7|th=%.2f|rank=%d|mode=%s|tp=%.3f|sl=%.3f|H=%d|n=%d|win=%.3f|mean=%.4f|med=%.4f|worst=%.4f|holdmed=%.1f|score=%.4f'%((threshold,rank)+cfg+(n,wr,mean,med,worst,hm,score)),flush=True)
# nested selection: for each OOS fold choose config using PRIOR OOS folds only, then apply to next fold. starts fold2.
for threshold in [.50,.55,.60]:
 allrets=[];chosen=[]
 for f in range(1,5):
  histmask=(idx>=D[int(len(D)*.50)])&(idx<D[int(len(D)*bounds[f])])&(pred>=threshold);test0=D[int(len(D)*bounds[f])];test1=D[-1]+pd.Timedelta(days=1) if f==4 else D[int(len(D)*bounds[f+1])];testmask=(idx>=test0)&(idx<test1)&(pred>=threshold);hist=t[histmask];test=t[testmask]
  if len(hist)<10 or len(test)==0:continue
  best=None
  for cfg in configs:
   rr=np.array([sim(r,*cfg)[0] for _,r in hist.iterrows()]);score=rr.mean()+.03*(rr>0).mean()+.15*rr.min();cand=(score,cfg)
   if best is None or cand[0]>best[0]:best=cand
  cfg=best[1];rr=np.array([sim(r,*cfg)[0] for _,r in test.iterrows()]);allrets.extend(rr.tolist());chosen.append(cfg);print('NEST7|th=%.2f|fold=%d|cfg=%s/%g/%g/%d|n=%d|win=%.3f|mean=%.4f|worst=%.4f'%(threshold,f+1,cfg[0],cfg[1],cfg[2],cfg[3],len(rr),(rr>0).mean(),rr.mean(),rr.min()),flush=True)
 if allrets:
  rr=np.array(allrets);print('NESTSUM7|th=%.2f|n=%d|win=%.3f|mean=%.4f|median=%.4f|worst=%.4f'%(threshold,len(rr),(rr>0).mean(),rr.mean(),np.median(rr),rr.min()),flush=True)
print('COST7|%.4f'%COST,flush=True);print('DONE7',flush=True)
