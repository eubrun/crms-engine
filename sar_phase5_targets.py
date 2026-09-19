"""Phase 5: walk-forward HGB meta-label; evaluate >=.60 signals for +10/+12/+15% target attainment before next bearish SAR flip."""
import warnings,numpy as np,pandas as pd
from crms import ASSETS,fetch,indicators,weekly
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings('ignore');rows=[]
for s in ASSETS:
 try:
  z=indicators(fetch(s)).join(weekly(fetch(s)));c=z.close;v=z.volume
  for n in [3,5,10,20,40,60,84]:z[f'r{n}']=c.pct_change(n)
  for n in [20,50,100,200]:z[f'ema{n}']=c.ewm(span=n,adjust=False).mean()
  z['atrp']=z.atr/c;z['di_gap']=z.plus_di-z.minus_di;z['bbw']=4*c.rolling(20).std()/c.rolling(20).mean();z['obv']=(np.sign(c.diff()).fillna(0)*v).cumsum();z['dd50']=c/c.rolling(50).max()-1
  bull=z.psar_bull.astype(bool);en=bull&~bull.shift(1,fill_value=False);ex=~bull&bull.shift(1,fill_value=False);E=np.flatnonzero(en);X=np.flatnonzero(ex);bear=~bull;age=bear.groupby(bear.ne(bear.shift()).cumsum()).cumcount()+1
  for i in E:
   q=[j for j in X if j>i]
   if not q or i<210:continue
   j=q[0];p=i-1;entry=float(c.iloc[i]);seg=z.iloc[i:j+1]
   def x(col,k=0):return float(z[col].iloc[p-k]) if pd.notna(z[col].iloc[p-k]) else np.nan
   r={'date':z.index[i],'symbol':s,'ret':float(c.iloc[j]/entry-1),'days':j-i,'mfe':float(seg.high.max()/entry-1),'mae':float(seg.low.min()/entry-1),'bear_age':float(age.iloc[p]),'weekly':int(bool(z.weekly_bull.iloc[i])) if pd.notna(z.weekly_bull.iloc[i]) else 0,'adx':x('adx'),'adx_d3':x('adx')-x('adx',3),'di':x('di_gap'),'di_d3':x('di_gap')-x('di_gap',3),'rsi':x('rsi'),'rsi_d3':x('rsi')-x('rsi',3),'hist':x('macd_hist'),'hist_d3':x('macd_hist')-x('macd_hist',3),'rvol':x('rvol20'),'atrp':x('atrp'),'bbw':x('bbw'),'bbw_d5':x('bbw')-x('bbw',5),'obv_d10':x('obv')-x('obv',10),'dd50':x('dd50')}
   for n in [5,10,20,40,60,84]:r[f'r{n}']=x(f'r{n}')
   for n in [20,50,100,200]:r[f'ema{n}gap']=x('close')/x(f'ema{n}')-1
   for target in [.10,.12,.15]:
    hits=np.flatnonzero((seg.high.to_numpy()/entry-1)>=target);r[f'hit{int(target*100)}']=int(len(hits)>0);r[f'day{int(target*100)}']=int(hits[0]) if len(hits) else np.nan
   rows.append(r)
 except Exception as e:print('FAIL|%s|%s'%(s,str(e)[:80]),flush=True)
t=pd.DataFrame(rows).set_index('date').sort_index();btc=indicators(fetch('BTCUSDT'));bc=btc.close;B=pd.DataFrame({'btc20':bc.pct_change(20),'btc60':bc.pct_change(60),'btc84':bc.pct_change(84),'btcrsi':btc.rsi,'btcadx':btc.adx,'btcdd50':bc/bc.rolling(50).max()-1},index=btc.index)
for k in B:t[k]=B[k].reindex(t.index).ffill().to_numpy()
t['rs20']=t.r20-t.btc20;t['rs60']=t.r60-t.btc60
exclude=['symbol','ret','days','mfe','mae','hit10','hit12','hit15','day10','day12','day15'];features=[c for c in t.columns if c not in exclude];X=t[features].astype(float);y=(t.ret>0).astype(int).to_numpy();idx=pd.DatetimeIndex(pd.to_datetime(t.index,utc=True)).tz_convert(None);D=idx.unique().sort_values()
# expanding walk-forward: first 50% training, then 5 chronological OOS folds
bounds=[.50,.60,.70,.80,.90,1.0];pred=np.full(len(t),np.nan)
for f in range(5):
 train_end=D[int(len(D)*bounds[f])]; test_end=D[-1]+pd.Timedelta(days=1) if f==4 else D[int(len(D)*bounds[f+1])];tr=idx<train_end;te=(idx>=train_end)&(idx<test_end)
 m=make_pipeline(SimpleImputer(),HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,learning_rate=.04,l2_regularization=3,min_samples_leaf=25,random_state=7));m.fit(X.loc[tr],y[tr]);pred[te]=m.predict_proba(X.loc[te])[:,1];print('FOLD|%d|train=%d|test=%d|%s..%s'%(f+1,tr.sum(),te.sum(),train_end.date(),test_end.date()),flush=True)
for th in [.55,.60,.65,.70]:
 q=(pred>=th);a=t[q];print('TH|%.2f|n=%d|win_exit=%.3f|ret_mean=%.4f|ret_med=%.4f|mfe=%.4f|mae=%.4f'%(th,len(a),(a.ret>0).mean(),a.ret.mean(),a.ret.median(),a.mfe.mean(),a.mae.mean()),flush=True)
 for target in [10,12,15]:print('TARGET|th=%.2f|t=%d|hit=%.3f|n=%d|days_hit_med=%.1f'%(th,target,a[f'hit{target}'].mean(),len(a),a.loc[a[f'hit{target}']==1,f'day{target}'].median()),flush=True)
# HGB >=.60 buckets and asset counts
q=pred>=.60;a=t[q].copy();a['prob']=pred[q];a['bucket']=pd.cut(a.prob,[.60,.65,.70,1.01],right=False,labels=['60-65','65-70','70+'])
for b,g in a.groupby('bucket',observed=True):print('BUCKET|%s|n=%d|win=%.3f|hit10=%.3f|hit12=%.3f|hit15=%.3f'%(b,len(g),(g.ret>0).mean(),g.hit10.mean(),g.hit12.mean(),g.hit15.mean()),flush=True)
print('ASSET60|'+';'.join('%s=%d'%(s,n) for s,n in a.symbol.value_counts().items()),flush=True);print('DONE',flush=True)
