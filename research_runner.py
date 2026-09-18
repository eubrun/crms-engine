"""Autonomous SAR research runner."""
from itertools import combinations
from pathlib import Path
import warnings,numpy as np,pandas as pd
from crms import ASSETS,fetch,indicators,weekly
warnings.filterwarnings('ignore');OUT=Path('output');OUT.mkdir(exist_ok=True)
def feats(s):
 d=fetch(s);z=indicators(d).join(weekly(d));c=z.close;v=z.volume
 for n in [3,5,10,20,40,60,84]:z[f'r{n}']=c.pct_change(n)
 for n in [10,20,50,100,200]:z[f'ema{n}']=c.ewm(span=n,adjust=False).mean()
 z['atrp']=z.atr/c;z['sar_gap']=(c-z.psar)/z.atr;z['di_gap']=z.plus_di-z.minus_di;z['bb_mid']=c.rolling(20).mean();sd=c.rolling(20).std();z['bb_width']=4*sd/z.bb_mid;z['obv']=(np.sign(c.diff()).fillna(0)*v).cumsum();mf=((c-z.low)-(z.high-c))/(z.high-z.low).replace(0,np.nan)*v;z['cmf20']=mf.rolling(20).sum()/v.rolling(20).sum();z['vol_z']=(v-v.rolling(20).mean())/v.rolling(20).std();z['dd50']=c/c.rolling(50).max()-1;z['ema_stack']=(c>z.ema20)&(z.ema20>z.ema50)&(z.ema50>z.ema200)
 bull=z.psar_bull.astype(bool);en=bull&~bull.shift(1,fill_value=False);ex=~bull&bull.shift(1,fill_value=False);E=np.flatnonzero(en);X=np.flatnonzero(ex);bear=~bull;age=bear.groupby(bear.ne(bear.shift()).cumsum()).cumcount()+1;rows=[]
 for i in E:
  q=[j for j in X if j>i]
  if not q or i<90:continue
  j=q[0];r={'date':z.index[i],'symbol':s,'ret':float(c.iloc[j]/c.iloc[i]-1),'days':j-i};val=lambda col,k=1:float(z[col].iloc[i-k]) if pd.notna(z[col].iloc[i-k]) else np.nan
  r.update({'bear_age':float(age.iloc[i-1]),'adx':val('adx'),'adx_d3':val('adx')-val('adx',4),'di_gap':val('di_gap'),'di_d3':val('di_gap')-val('di_gap',4),'minus_d3':val('minus_di')-val('minus_di',4),'rsi':val('rsi'),'rsi_d3':val('rsi')-val('rsi',4),'hist':val('macd_hist'),'hist_d3':val('macd_hist')-val('macd_hist',4),'hist_d7':val('macd_hist')-val('macd_hist',8),'rvol':val('rvol20'),'vol_z':val('vol_z'),'atrp':val('atrp'),'atr_d5':val('atrp')-val('atrp',6),'bbw':val('bb_width'),'bbw_d5':val('bb_width')-val('bb_width',6),'cmf':val('cmf20'),'obv_d10':val('obv')-val('obv',11),'sar_gap':val('sar_gap'),'dd50':val('dd50'),'weekly_flag':bool(z.weekly_bull.iloc[i]) if pd.notna(z.weekly_bull.iloc[i]) else False,'stack_flag':bool(z.ema_stack.iloc[i-1])})
  for n in [5,10,20,40,60,84]:r[f'r{n}']=val(f'r{n}')
  rows.append(r)
 return rows
R=[]
for s in ASSETS:
 try:R+=feats(s);print('LOAD|'+s,flush=True)
 except Exception as e:print('FAIL|%s|%s'%(s,str(e)[:100]),flush=True)
t=pd.DataFrame(R).set_index('date').sort_index();btc=indicators(fetch('BTCUSDT'));bc=btc.close;B=pd.DataFrame(index=btc.index);B['btc_r20']=bc.pct_change(20);B['btc_r60']=bc.pct_change(60);B['btc_dd50']=bc/bc.rolling(50).max()-1;B['btc_rsi']=btc.rsi;B['btc_adx']=btc.adx
for k in B:t[k]=B[k].reindex(t.index).ffill().to_numpy()
t['rs20v']=t['r20']-t['btc_r20'];t['rs60v']=t['r60']-t['btc_r60'];t['rank20']=t.groupby(level=0)['r20'].rank(pct=True);t['rank60']=t.groupby(level=0)['r60'].rank(pct=True)
C={'weekly':t['weekly_flag'],'stack':t['stack_flag'],'bear5':t['bear_age']>=5,'bear10':t['bear_age']>=10,'bear20':t['bear_age']>=20,'adx15_30':t['adx'].between(15,30),'adx20_35':t['adx'].between(20,35),'adxrise':t['adx_d3']>0,'di_pos':t['di_gap']>0,'di_improve':t['di_d3']>0,'minusfall':t['minus_d3']<0,'rsi45_60':t['rsi'].between(45,60),'rsi50_65':t['rsi'].between(50,65),'rsirise':t['rsi_d3']>0,'histneg_improve':(t['hist']<0)&(t['hist_d3']>0),'histimprove':t['hist_d3']>0,'hist7':t['hist_d7']>0,'rvol_07_15':t['rvol'].between(.7,1.5),'rvol_gt1':t['rvol']>1,'volz_lt1':t['vol_z']<1,'atrcompress':t['atr_d5']<0,'atrexpand':t['atr_d5']>0,'bbsqueeze':t['bbw_d5']<0,'bbexpand':t['bbw_d5']>0,'cmfpos':t['cmf']>0,'obvup':t['obv_d10']>0,'ddmild':t['dd50']>-.20,'m5':t['r5']>0,'m10':t['r10']>0,'m20':t['r20']>0,'m40':t['r40']>0,'slow60':t['r60']>0,'slow84':t['r84']>0,'rs20':t['rs20v']>0,'rs60':t['rs60v']>0,'rank20top':t['rank20']>=.7,'rank60top':t['rank60']>=.7,'btcmom20':t['btc_r20']>0,'btcmom60':t['btc_r60']>0,'btcrsi50':t['btc_rsi']>=50,'btcadx20':t['btc_adx']>=20,'btcddmild':t['btc_dd50']>-.15}
names=list(C);A=np.column_stack([C[k].fillna(False).to_numpy(dtype=bool) for k in names]);y=t['ret'].to_numpy(float);idx=pd.DatetimeIndex(pd.to_datetime(t.index,utc=True)).tz_convert(None);D=idx.unique().sort_values();c1=D[int(len(D)*.55)];c2=D[int(len(D)*.75)];P={'tr':idx<c1,'va':(idx>=c1)&(idx<c2),'te':idx>=c2}
def st(m,p):
 q=m&np.asarray(P[p]);n=int(q.sum());return n,float((y[q]>0).mean()) if n else 0,float(y[q].mean()) if n else 0,float(np.median(y[q])) if n else 0
leader=[];checked=0
for k in range(1,6):
 for ids in combinations(range(len(names)),k):
  checked+=1;m=A[:,ids].all(1);n,w,_,_=st(m,'tr');nv,wv,_,_=st(m,'va')
  if n<60 or nv<20 or min(w,wv)<.50:continue
  nt,wt,mut,mdt=st(m,'te')
  if nt>=20:leader.append({'rule':'+'.join(names[i] for i in ids),'k':k,'train_n':n,'train_win':w,'valid_n':nv,'valid_win':wv,'test_n':nt,'test_win':wt,'test_mean':mut,'test_median':mdt,'robust_min':min(w,wv,wt)})
 print('PROGRESS|k=%d|checked=%d|kept=%d'%(k,checked,len(leader)),flush=True)
L=pd.DataFrame(leader)
if len(L):L=L.sort_values(['robust_min','test_win','test_mean'],ascending=False)
L.to_csv(OUT/'sar_v2_leaderboard.csv',index=False);print('BASE|n=%d|win=%.4f|mean=%.4f|days=%.2f|split=%s,%s'%(len(t),(y>0).mean(),y.mean(),t['days'].mean(),c1.date(),c2.date()),flush=True)
for _,r in L.head(50).iterrows():print('RULE|%s|tr=%.3f/%d|va=%.3f/%d|te=%.3f/%d|mean=%.4f|med=%.4f|robust=%.3f'%(r.rule,r.train_win,r.train_n,r.valid_win,r.valid_n,r.test_win,r.test_n,r.test_mean,r.test_median,r.robust_min),flush=True)
print('DONE|checked=%d|kept=%d'%(checked,len(L)),flush=True)
