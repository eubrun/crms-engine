"""Phase 4: meta-label bullish daily PSAR flips. Strict chronological train/validation/OOS; probability thresholds trade frequency for precision."""
import warnings,numpy as np,pandas as pd
from crms import ASSETS,fetch,indicators,weekly
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import roc_auc_score
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
   j=q[0];p=i-1
   def x(col,k=0):return float(z[col].iloc[p-k]) if pd.notna(z[col].iloc[p-k]) else np.nan
   r={'date':z.index[i],'symbol':s,'ret':float(c.iloc[j]/c.iloc[i]-1),'days':j-i,'bear_age':float(age.iloc[p]),'weekly':int(bool(z.weekly_bull.iloc[i])) if pd.notna(z.weekly_bull.iloc[i]) else 0,'adx':x('adx'),'adx_d3':x('adx')-x('adx',3),'di':x('di_gap'),'di_d3':x('di_gap')-x('di_gap',3),'rsi':x('rsi'),'rsi_d3':x('rsi')-x('rsi',3),'hist':x('macd_hist'),'hist_d3':x('macd_hist')-x('macd_hist',3),'rvol':x('rvol20'),'atrp':x('atrp'),'bbw':x('bbw'),'bbw_d5':x('bbw')-x('bbw',5),'obv_d10':x('obv')-x('obv',10),'dd50':x('dd50')}
   for n in [5,10,20,40,60,84]:r[f'r{n}']=x(f'r{n}')
   for n in [20,50,100,200]:r[f'ema{n}gap']=x('close')/x(f'ema{n}')-1
   rows.append(r)
 except Exception as e:print('FAIL|%s|%s'%(s,str(e)[:90]),flush=True)
t=pd.DataFrame(rows).set_index('date').sort_index();btc=indicators(fetch('BTCUSDT'));bc=btc.close
B=pd.DataFrame({'btc20':bc.pct_change(20),'btc60':bc.pct_change(60),'btc84':bc.pct_change(84),'btcrsi':btc.rsi,'btcadx':btc.adx,'btcdd50':bc/bc.rolling(50).max()-1},index=btc.index)
for k in B:t[k]=B[k].reindex(t.index).ffill().to_numpy()
t['rs20']=t.r20-t.btc20;t['rs60']=t.r60-t.btc60
features=[c for c in t.columns if c not in ['symbol','ret','days']];X=t[features].astype(float);y=(t.ret>0).astype(int).to_numpy();rets=t.ret.to_numpy();idx=pd.DatetimeIndex(pd.to_datetime(t.index,utc=True)).tz_convert(None);D=idx.unique().sort_values();c1=D[int(len(D)*.55)];c2=D[int(len(D)*.75)];tr=idx<c1;va=(idx>=c1)&(idx<c2);te=idx>=c2
models={'LOG':make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(C=.2,class_weight='balanced',max_iter=2000)),'HGB':make_pipeline(SimpleImputer(),HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,learning_rate=.04,l2_regularization=3,min_samples_leaf=25,random_state=7)),'RF':make_pipeline(SimpleImputer(),RandomForestClassifier(n_estimators=500,max_depth=4,min_samples_leaf=15,max_features=.6,class_weight='balanced',random_state=7,n_jobs=-1))}
print('DATA|n=%d|base=%.3f|split=%s,%s|features=%d'%(len(t),y.mean(),c1.date(),c2.date(),len(features)),flush=True)
for name,m in models.items():
 m.fit(X.loc[tr],y[tr]);pv=m.predict_proba(X.loc[va])[:,1];pt=m.predict_proba(X.loc[te])[:,1]
 try:aucv=roc_auc_score(y[va],pv);auct=roc_auc_score(y[te],pt)
 except:aucv=auct=np.nan
 print('MODEL|%s|auc_va=%.3f|auc_te=%.3f'%(name,aucv,auct),flush=True)
 # threshold selected ONLY by validation; require >=20 validation signals. Then report untouched OOS.
 candidates=[]
 for th in np.arange(.45,.91,.01):
  q=pv>=th;n=q.sum()
  if n>=20:candidates.append(((y[va][q]).mean(),float(th),int(n),rets[va][q].mean()))
 candidates.sort(reverse=True);best=candidates[0] if candidates else (0,.5,0,0);_,th,nv,_=best
 for label,p,mask in [('VA',pv,va),('OOS',pt,te)]:
  q=p>=th; yy=y[mask][q];rr=rets[mask][q]
  print('PICK|%s|%s|th=%.2f|n=%d|win=%.3f|mean=%.4f|median=%.4f'%(name,label,th,len(yy),yy.mean() if len(yy) else 0,rr.mean() if len(rr) else 0,np.median(rr) if len(rr) else 0),flush=True)
 # fixed confidence grid prevents cherry-picking OOS
 for th0 in [.50,.55,.60,.65,.70,.75,.80]:
  q=pt>=th0;yy=y[te][q];rr=rets[te][q]
  print('GRID|%s|th=%.2f|n=%d|win=%.3f|mean=%.4f|med=%.4f'%(name,th0,len(yy),yy.mean() if len(yy) else 0,rr.mean() if len(rr) else 0,np.median(rr) if len(rr) else 0),flush=True)
print('DONE',flush=True)
