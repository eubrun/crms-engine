"""Phase 2: characterize explosive SAR winners vs failures using only pre-entry information."""
import warnings,numpy as np,pandas as pd
from crms import ASSETS,fetch,indicators,weekly
warnings.filterwarnings('ignore');rows=[]
for s in ASSETS:
 try:
  z=indicators(fetch(s)).join(weekly(fetch(s)));c=z.close;v=z.volume
  z['atrp']=z.atr/c;z['dip']=z.plus_di-z.minus_di;z['ema20']=c.ewm(span=20,adjust=False).mean();z['ema50']=c.ewm(span=50,adjust=False).mean();z['ema200']=c.ewm(span=200,adjust=False).mean();z['bbw']=4*c.rolling(20).std()/c.rolling(20).mean();z['obv']=(np.sign(c.diff()).fillna(0)*v).cumsum();z['dd50']=c/c.rolling(50).max()-1
  for n in [5,10,20,40,60,84]:z[f'r{n}']=c.pct_change(n)
  bull=z.psar_bull.astype(bool);en=bull&~bull.shift(1,fill_value=False);ex=~bull&bull.shift(1,fill_value=False);E=np.flatnonzero(en);X=np.flatnonzero(ex);bear=~bull;age=bear.groupby(bear.ne(bear.shift()).cumsum()).cumcount()+1
  for i in E:
   q=[j for j in X if j>i]
   if not q or i<90:continue
   j=q[0];p=i-1
   def x(col,k=0):return float(z[col].iloc[p-k]) if pd.notna(z[col].iloc[p-k]) else np.nan
   r={'date':z.index[i],'symbol':s,'ret':float(c.iloc[j]/c.iloc[i]-1),'days':j-i,'bear_age':age.iloc[p],'adx':x('adx'),'adx_d3':x('adx')-x('adx',3),'di':x('dip'),'di_d3':x('dip')-x('dip',3),'rsi':x('rsi'),'rsi_d3':x('rsi')-x('rsi',3),'hist':x('macd_hist'),'hist_d3':x('macd_hist')-x('macd_hist',3),'hist_d7':x('macd_hist')-x('macd_hist',7),'rvol':x('rvol20'),'atrp':x('atrp'),'atr_d5':x('atrp')-x('atrp',5),'bbw':x('bbw'),'bbw_d5':x('bbw')-x('bbw',5),'obv_d10':x('obv')-x('obv',10),'dd50':x('dd50'),'ema20gap':x('close')/x('ema20')-1,'ema50gap':x('close')/x('ema50')-1,'ema200gap':x('close')/x('ema200')-1,'weekly':int(bool(z.weekly_bull.iloc[i])) if pd.notna(z.weekly_bull.iloc[i]) else 0}
   for n in [5,10,20,40,60,84]:r[f'r{n}']=x(f'r{n}')
   rows.append(r)
 except Exception as e:print('FAIL|%s|%s'%(s,str(e)[:80]),flush=True)
t=pd.DataFrame(rows).set_index('date').sort_index();btc=indicators(fetch('BTCUSDT'));bc=btc.close;B=pd.DataFrame({'btc20':bc.pct_change(20),'btc60':bc.pct_change(60),'btcrsi':btc.rsi},index=btc.index)
for k in B:t[k]=B[k].reindex(t.index).ffill().to_numpy()
t['rs20']=t.r20-t.btc20;t['rs60']=t.r60-t.btc60
features=[c for c in t.columns if c not in ['symbol','ret','days']];idx=pd.DatetimeIndex(pd.to_datetime(t.index,utc=True)).tz_convert(None);D=idx.unique().sort_values();cut=D[int(len(D)*.75)];parts={'PRE':idx<cut,'OOS':idx>=cut}
print('DATA|n=%d|win=%.3f|mean=%.4f|cut=%s'% (len(t),(t.ret>0).mean(),t.ret.mean(),cut.date()),flush=True)
for part,pm in parts.items():
 a=t[np.asarray(pm)];print('PART|%s|n=%d|win=%.3f|big10=%d|big20=%d|fail=%d'%(part,len(a),(a.ret>0).mean(),(a.ret>=.10).sum(),(a.ret>=.20).sum(),(a.ret<=0).sum()),flush=True)
 # standardized median effect explosive >=10% versus failure <=0
 for target,thr in [('BIG10',.10),('BIG20',.20)]:
  W=a[a.ret>=thr];F=a[a.ret<=0];out=[]
  for f in features:
   if f=='weekly':continue
   sd=a[f].std()
   if pd.notna(sd) and sd>0:out.append((abs((W[f].median()-F[f].median())/sd),(W[f].median()-F[f].median())/sd,f,W[f].median(),F[f].median()))
  print('TOP|%s|%s|'% (part,target)+';'.join('%s:effect=%.2f,w=%.4g,f=%.4g'%(f,e,w,m) for _,e,f,w,m in sorted(out,reverse=True)[:15]),flush=True)
 # asset distribution of explosive winners
 print('ASSETS|%s|BIG10|'%part+';'.join('%s=%d/%d'%(s,((a.symbol==s)&(a.ret>=.10)).sum(),(a.symbol==s).sum()) for s in sorted(a.symbol.unique())),flush=True)
print('DONE',flush=True)
