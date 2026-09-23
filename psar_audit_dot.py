import requests,pandas as pd,numpy as np
URL='https://data-api.binance.vision/api/v3/klines'
def get(symbol):
 r=requests.get(URL,params={'symbol':symbol,'interval':'1w','limit':1000},timeout=30);r.raise_for_status();x=r.json();d=pd.DataFrame(x,columns=['ot','o','h','l','c','v','ct','q','n','tb','tq','i'])
 for col in ['o','h','l','c']: d[col]=pd.to_numeric(d[col])
 d['date']=pd.to_datetime(d.ot,unit='ms',utc=True);return d

def psar(d,inclusive=False,start=.02,inc=.02,maxaf=.2):
 h=d.h.to_numpy();l=d.l.to_numpy();n=len(d);sar=np.full(n,np.nan);raw=np.full(n,np.nan);trend=np.zeros(n,dtype=int);ep=np.full(n,np.nan);af=np.full(n,np.nan);rev=np.zeros(n,dtype=bool)
 bull=d.c.iloc[1]>=d.c.iloc[0];sar[1]=l[0] if bull else h[0];ep[1]=h[1] if bull else l[1];af[1]=start;trend[1]=1 if bull else -1
 for i in range(2,n):
  up=trend[i-1]==1;q=sar[i-1]+af[i-1]*(ep[i-1]-sar[i-1]);raw[i]=q;a=af[i-1];e=ep[i-1]
  hit=(q>=l[i]) if (up and inclusive) else ((q>l[i]) if up else ((q<=h[i]) if inclusive else (q<h[i])))
  if up:
   if hit: bull=False;rev[i]=1;q=e;e=l[i];a=start
   else:
    bull=True
    if h[i]>e:e=h[i];a=min(maxaf,a+inc)
    q=min(q,l[i-1],l[i-2])
  else:
   if hit: bull=True;rev[i]=1;q=e;e=h[i];a=start
   else:
    bull=False
    if l[i]<e:e=l[i];a=min(maxaf,a+inc)
    q=max(q,h[i-1],h[i-2])
  sar[i]=q;ep[i]=e;af[i]=a;trend[i]=1 if bull else -1
 return pd.DataFrame({'raw':raw,'sar':sar,'trend':trend,'ep':ep,'af':af,'rev':rev})

d=get('DOTUSDC')
for inclusive in [False,True]:
 a=psar(d,inclusive);z=pd.concat([d[['date','h','l','c']],a],axis=1);mode='INCLUSIVE' if inclusive else 'STRICT'
 print('COMPARE_BEGIN|mode='+mode);print(z.tail(8).to_string(index=False));r=z.iloc[-1]
 print('COMPARE_RESULT|mode=%s|date=%s|close=%.8g|sar=%.8g|trend=%s|ep=%.8g|af=%.2f|rev=%d'%(mode,r.date,r.c,r.sar,'BULL' if r.trend==1 else 'BEAR',r.ep,r.af,r.rev))
