import requests,pandas as pd,numpy as np
BYBIT='https://api.bybit.com/v5/market/kline'
def get_bybit():
 r=requests.get(BYBIT,params={'category':'spot','symbol':'DOTUSDC','interval':'W','limit':1000},timeout=30);r.raise_for_status();j=r.json()
 if j.get('retCode')!=0: raise RuntimeError(j)
 x=j['result']['list'];d=pd.DataFrame(x,columns=['ot','o','h','l','c','v','turnover']);d=d.iloc[::-1].reset_index(drop=True)
 for col in ['o','h','l','c']: d[col]=pd.to_numeric(d[col])
 d['date']=pd.to_datetime(pd.to_numeric(d.ot),unit='ms',utc=True);return d

def psar(d,start=.02,inc=.02,maxaf=.2):
 h=d.h.to_numpy();l=d.l.to_numpy();n=len(d);sar=np.full(n,np.nan);raw=np.full(n,np.nan);trend=np.zeros(n,dtype=int);ep=np.full(n,np.nan);af=np.full(n,np.nan);rev=np.zeros(n,dtype=bool)
 bull=d.c.iloc[1]>=d.c.iloc[0];sar[1]=l[0] if bull else h[0];ep[1]=h[1] if bull else l[1];af[1]=start;trend[1]=1 if bull else -1
 for i in range(2,n):
  up=trend[i-1]==1;q=sar[i-1]+af[i-1]*(ep[i-1]-sar[i-1]);raw[i]=q;a=af[i-1];e=ep[i-1]
  if up:
   if q>l[i]: bull=False;rev[i]=1;q=e;e=l[i];a=start
   else:
    bull=True
    if h[i]>e:e=h[i];a=min(maxaf,a+inc)
    q=min(q,l[i-1],l[i-2])
  else:
   if q<h[i]: bull=True;rev[i]=1;q=e;e=h[i];a=start
   else:
    bull=False
    if l[i]<e:e=l[i];a=min(maxaf,a+inc)
    q=max(q,h[i-1],h[i-2])
  sar[i]=q;ep[i]=e;af[i]=a;trend[i]=1 if bull else -1
 return pd.DataFrame({'raw':raw,'sar':sar,'trend':trend,'ep':ep,'af':af,'rev':rev})

d=get_bybit();a=psar(d);z=pd.concat([d[['date','o','h','l','c']],a],axis=1)
print('BYBIT_AUDIT|bars=%d|first=%s|last=%s'%(len(z),z.iloc[0].date,z.iloc[-1].date))
print('BYBIT_RECENT_BEGIN');print(z.tail(12).to_string(index=False));print('BYBIT_RECENT_END')
print('BYBIT_REVERSALS_BEGIN')
for i,r in z[z.rev].tail(10).iterrows():
 p=z.iloc[i-1];print('BYBIT_REV|date=%s|to=%s|H=%.8g|L=%.8g|raw=%.8g|sar=%.8g|prevSAR=%.8g|prevEP=%.8g|prevAF=%.2f|newEP=%.8g'%(r.date,'BULL' if r.trend==1 else 'BEAR',r.h,r.l,r.raw,r.sar,p.sar,p.ep,p.af,r.ep))
r=z.iloc[-1];print('BYBIT_RESULT|date=%s|close=%.8g|H=%.8g|L=%.8g|raw=%.8g|sar=%.8g|trend=%s|ep=%.8g|af=%.2f|rev=%d'%(r.date,r.c,r.h,r.l,r.raw,r.sar,'BULL' if r.trend==1 else 'BEAR',r.ep,r.af,r.rev))
