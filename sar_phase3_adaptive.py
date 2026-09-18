"""Phase 3: regime-adaptive percentile filters for bullish daily PSAR flips."""
import warnings,numpy as np,pandas as pd
from itertools import combinations
from crms import ASSETS,fetch,indicators,weekly
warnings.filterwarnings('ignore');rows=[]
for s in ASSETS:
 try:
  z=indicators(fetch(s)).join(weekly(fetch(s)));c=z.close;v=z.volume
  for n in [20,60,84]:z[f'r{n}']=c.pct_change(n)
  z['ema200']=c.ewm(span=200,adjust=False).mean();z['ema200gap']=c/z.ema200-1;z['atrp']=z.atr/c
  bull=z.psar_bull.astype(bool);en=bull&~bull.shift(1,fill_value=False);ex=~bull&bull.shift(1,fill_value=False);E=np.flatnonzero(en);X=np.flatnonzero(ex);bear=~bull;age=bear.groupby(bear.ne(bear.shift()).cumsum()).cumcount()+1
  # rolling percentiles computed from prior data only
  for col in ['adx','rvol20','rsi','atrp','ema200gap']:
   z[col+'_pct']=z[col].shift(1).rolling(252,min_periods=100).rank(pct=True)
  for i in E:
   q=[j for j in X if j>i]
   if not q or i<252:continue
   j=q[0];p=i-1
   r={'date':z.index[i],'symbol':s,'ret':float(c.iloc[j]/c.iloc[i]-1),'days':j-i,'bear_age':float(age.iloc[p]),'weekly':bool(z.weekly_bull.iloc[i]) if pd.notna(z.weekly_bull.iloc[i]) else False}
   for col in ['adx','rvol20','rsi','atrp','ema200gap']:r[col+'p']=float(z[col+'_pct'].iloc[i]) if pd.notna(z[col+'_pct'].iloc[i]) else np.nan
   r['r20']=float(z.r20.iloc[p]);r['r60']=float(z.r60.iloc[p]);r['r84']=float(z.r84.iloc[p]);rows.append(r)
 except Exception as e:print('FAIL|%s|%s'%(s,str(e)[:80]),flush=True)
t=pd.DataFrame(rows).set_index('date').sort_index();btc=indicators(fetch('BTCUSDT'));bc=btc.close;B=pd.DataFrame({'btc20':bc.pct_change(20),'btc60':bc.pct_change(60),'btcrsi':btc.rsi},index=btc.index)
for k in B:t[k]=B[k].reindex(t.index).ffill().to_numpy()
t['rs60']=t.r60-t.btc60
# adaptive hypotheses; thresholds are coarse to test stability, not optimize exact decimals
C={}
for q in [.2,.3,.4,.5,.6,.7,.8]:
 C[f'adx_le{int(q*10)}']=t.adxp<=q;C[f'rvol_ge{int(q*10)}']=t.rvol20p>=q;C[f'rsi_ge{int(q*10)}']=t.rsip>=q;C[f'ema200_ge{int(q*10)}']=t.ema200gapp>=q;C[f'atr_ge{int(q*10)}']=t.atrpp>=q
C.update({'bear10':t.bear_age>=10,'bear15':t.bear_age>=15,'weekly':t.weekly,'btc60pos':t.btc60>0,'btc60_5':t.btc60>.05,'btc60_10':t.btc60>.10,'btcrsi50':t.btcrsi>=50,'btcrsi55':t.btcrsi>=55,'rs60pos':t.rs60>0,'m60pos':t.r60>0,'m84pos':t.r84>0})
names=list(C);A=np.column_stack([C[k].fillna(False).to_numpy(bool) for k in names]);y=t.ret.to_numpy(float);idx=pd.DatetimeIndex(pd.to_datetime(t.index,utc=True)).tz_convert(None);D=idx.unique().sort_values();c1=D[int(len(D)*.55)];c2=D[int(len(D)*.75)];P={'tr':idx<c1,'va':(idx>=c1)&(idx<c2),'te':idx>=c2}
def st(m,p):
 q=m&np.asarray(P[p]);n=int(q.sum());return n,float((y[q]>0).mean()) if n else 0,float(y[q].mean()) if n else 0,float(np.median(y[q])) if n else 0
L=[];checked=0
for k in range(1,5):
 for ids in combinations(range(len(names)),k):
  checked+=1;m=A[:,ids].all(1);n,w,_,_=st(m,'tr');nv,wv,_,_=st(m,'va')
  if n<50 or nv<18 or min(w,wv)<.48:continue
  nt,wt,mu,md=st(m,'te')
  if nt>=18:L.append((min(w,wv,wt),wt,mu,nt,w,wv,ids,md))
 print('PROGRESS|k=%d|checked=%d|kept=%d'%(k,checked,len(L)),flush=True)
L.sort(reverse=True);print('BASE|n=%d|win=%.3f|mean=%.4f|days=%.2f|split=%s,%s'%(len(t),(y>0).mean(),y.mean(),t.days.mean(),c1.date(),c2.date()),flush=True)
for robust,wt,mu,nt,w,wv,ids,md in L[:60]:print('RULE|%s|tr=%.3f|va=%.3f|te=%.3f/%d|mean=%.4f|med=%.4f|robust=%.3f'%('+'.join(names[i] for i in ids),w,wv,wt,nt,mu,md,robust),flush=True)
print('DONE|checked=%d|kept=%d'%(checked,len(L)),flush=True)
