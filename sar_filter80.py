from itertools import combinations
import warnings, numpy as np, pandas as pd
from crms import ASSETS,fetch,indicators,weekly
warnings.filterwarnings('ignore')
rows=[]
for s in ASSETS:
 try:
  d=fetch(s);z=indicators(d).join(weekly(d));c=z.close
  for n in [5,10,20]:z[f'r{n}']=c.pct_change(n)
  for n in [20,50,200]:z[f'e{n}']=c.ewm(span=n,adjust=False).mean()
  z['ema_stack']=(c>z.e20)&(z.e20>z.e50)&(z.e50>z.e200);z['adxup']=z.adx>z.adx.shift(3);z['dip']=z.plus_di-z.minus_di;z['rsiup']=z.rsi>z.rsi.shift(3);z['atrp']=z.atr/c;z['atrexp']=z.atrp>z.atrp.shift(5);z['break20']=c>=z.high.shift().rolling(20).max();z['break50']=c>=z.high.shift().rolling(50).max()
  bull=z.psar_bull.astype(bool);en=bull&~bull.shift(1,fill_value=False);ex=~bull&bull.shift(1,fill_value=False);E=np.flatnonzero(en.to_numpy());X=np.flatnonzero(ex.to_numpy())
  for i in E:
   q=[j for j in X if j>i]
   if not q:continue
   j=q[0];r=z.iloc[i].copy();r['symbol']=s;r['trade_ret']=float(c.iloc[j]/c.iloc[i]-1);r['days']=j-i;rows.append(r)
 except Exception as e:print('FAIL|'+s+'|'+str(e)[:80],flush=True)
t=pd.DataFrame(rows).sort_index(); btc=indicators(fetch('BTCUSDT'));bc=btc.close;be20=bc.ewm(span=20,adjust=False).mean();be50=bc.ewm(span=50,adjust=False).mean();be200=bc.ewm(span=200,adjust=False).mean();B=pd.DataFrame({'btcbull':(be20>be50)&(be50>be200),'btcmom20':bc.pct_change(20)>0,'btcrsi50':btc.rsi>=50})
for k in B:t[k]=B[k].reindex(t.index).ffill().to_numpy()
C={'weekly':t.weekly_bull.eq(True),'ema_stack':t.ema_stack,'btcbull':t.btcbull,'btcmom20':t.btcmom20,'btcrsi50':t.btcrsi50,'macdpos':t.macd_hist>0,'macdup':t.macd_up,'dip':t.dip>0,'di10':t.dip>10,'adx20':t.adx>=20,'adx25':t.adx>=25,'adxup':t.adxup,'rsi50':t.rsi>=50,'rsi5570':(t.rsi>=55)&(t.rsi<=70),'rsiup':t.rsiup,'rv1':t.rvol20>=1,'rv15':t.rvol20>=1.5,'m5pos':t.r5>0,'m5_2':t.r5>.02,'m10_5':t.r10>.05,'m20_10':t.r20>.10,'break20':t.break20,'break50':t.break50,'atrexp':t.atrexp}
D=pd.DatetimeIndex(pd.to_datetime(t.index,utc=True)).tz_convert(None).unique().sort_values();c1=D[int(len(D)*.55)];c2=D[int(len(D)*.75)];idx=pd.DatetimeIndex(pd.to_datetime(t.index,utc=True)).tz_convert(None);parts={'train':idx<c1,'valid':(idx>=c1)&(idx<c2),'test':idx>=c2};A=np.column_stack([C[k].fillna(False).to_numpy(bool) for k in C]);names=list(C);y=t.trade_ret.to_numpy(float)
def st(m,p):
 q=m&np.asarray(parts[p]);n=q.sum();return int(n),float((y[q]>0).mean()) if n else 0,float(y[q].mean()) if n else 0,float(np.median(y[q])) if n else 0
print('BASE|trades=%d|win=%.4f|mean=%.4f|median=%.4f|avgdays=%.2f|split=%s,%s'%(len(t),(y>0).mean(),y.mean(),np.median(y),t.days.mean(),c1.date(),c2.date()),flush=True)
cand=[];checked=0
for k in range(1,7):
 for ids in combinations(range(len(names)),k):
  checked+=1;m=A[:,ids].all(1);n,w,mu,md=st(m,'train');nv,wv,mv,mdv=st(m,'valid')
  if n>=80 and nv>=25:cand.append((min(w,wv),n+nv,ids,w,wv))
 print('PROGRESS|k=%d|checked=%d|cand=%d'%(k,checked,len(cand)),flush=True)
cand.sort(reverse=True);shown=0
for pre,npre,ids,w,wv in cand:
 m=A[:,ids].all(1);nt,wt,mt,mdt=st(m,'test')
 if nt<20:continue
 print('RULE|%s|train=%.4f|valid=%.4f|test_n=%d|test=%.4f|mean=%.4f|median=%.4f'%('+'.join(names[i] for i in ids),w,wv,nt,wt,mt,mdt),flush=True);shown+=1
 if shown>=30:break
print('DONE|checked=%d'%checked,flush=True)
