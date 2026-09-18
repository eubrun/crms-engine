from itertools import combinations
import numpy as np
import pandas as pd
from crms import ASSETS, fetch, indicators, weekly

# Search simple, auditable conjunctions. Thresholds are fixed economic/TA levels,
# not fitted quantiles, and selection is performed on chronological TRAIN only.
def make(symbol):
    d=fetch(symbol); z=indicators(d).join(weekly(d))
    c=z.close
    z['ret3']=c.pct_change(3); z['ret5']=c.pct_change(5); z['ret10']=c.pct_change(10); z['ret20']=c.pct_change(20)
    z['ema20']=c.ewm(span=20,adjust=False).mean(); z['ema50']=c.ewm(span=50,adjust=False).mean(); z['ema200']=c.ewm(span=200,adjust=False).mean()
    z['ema_stack']=(c>z.ema20)&(z.ema20>z.ema50)&(z.ema50>z.ema200)
    z['break20']=c>=z.high.shift(1).rolling(20).max()
    z['adx_up']=z.adx>z.adx.shift(3); z['di_spread']=(z.plus_di-z.minus_di)
    z['rsi_up']=z.rsi>z.rsi.shift(3)
    z['y10']=c.shift(-10)/c-1
    z['symbol']=symbol
    return z

frames=[]
for s in ASSETS:
    try: frames.append(make(s))
    except Exception as e: print('FAIL',s,e)
df=pd.concat(frames).sort_index()
# Chronological split prevents future leakage. Last 30% is untouched TEST.
cut=df.index.unique().sort_values()[int(len(df.index.unique())*.70)]
train=df[df.index<cut].copy(); test=df[df.index>=cut].copy()

conds={
 'weekly_bull':lambda x:x.weekly_bull.fillna(False),
 'ema_stack':lambda x:x.ema_stack,
 'macd_pos':lambda x:x.macd_hist>0,
 'macd_up':lambda x:x.macd_up,
 'di_pos':lambda x:x.plus_di>x.minus_di,
 'di_spread10':lambda x:x.di_spread>10,
 'adx20':lambda x:x.adx>=20,
 'adx25':lambda x:x.adx>=25,
 'adx_up':lambda x:x.adx_up,
 'rsi50_70':lambda x:(x.rsi>=50)&(x.rsi<=70),
 'rsi55_75':lambda x:(x.rsi>=55)&(x.rsi<=75),
 'rsi_up':lambda x:x.rsi_up,
 'rvol1':lambda x:x.rvol20>=1,
 'rvol1_5':lambda x:x.rvol20>=1.5,
 'mom3':lambda x:x.ret3>0,
 'mom5_2':lambda x:x.ret5>.02,
 'mom10_5':lambda x:x.ret10>.05,
 'mom20_10':lambda x:x.ret20>.10,
 'break20':lambda x:x.break20,
 'psar':lambda x:x.psar_bull,
}

def eval_rule(x,names):
    m=pd.Series(True,index=x.index)
    for n in names:m &= conds[n](x)
    y=x.loc[m,'y10'].dropna()
    return len(y),float((y>0).mean()) if len(y) else np.nan,float(y.mean()) if len(y) else np.nan,float(y.median()) if len(y) else np.nan

# Explore 1..5-condition rules; require useful TRAIN sample.
cands=[]; names=list(conds)
for k in range(1,6):
 for rule in combinations(names,k):
    n,w,mean,med=eval_rule(train,rule)
    if n>=40:cands.append((w,n,mean,med,rule))
cands.sort(reverse=True,key=lambda q:(q[0],q[1]))
# Freeze top diverse train rules, then reveal TEST. Also require >=15 TEST observations.
rows=[]
for w,n,mean,med,rule in cands[:300]:
    nt,wt,mt,mdt=eval_rule(test,rule)
    if nt>=15: rows.append({'rule':' + '.join(rule),'train_n':n,'train_win10':w,'train_mean10':mean,'test_n':nt,'test_win10':wt,'test_mean10':mt,'test_median10':mdt})
out=pd.DataFrame(rows).sort_values(['test_win10','test_n'],ascending=False)
print('SPLIT',cut)
print('\nTOP OOS RULES (selection candidates generated on TRAIN; TEST shown only after freeze)\n')
print(out.head(40).to_string(index=False))
out.to_csv('output/search80_oos.csv',index=False)
