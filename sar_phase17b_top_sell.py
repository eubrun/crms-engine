"""Phase17B: focused trailing-stop analysis of BUY -> TOP -> SELL timing.
Same Phase13B event universe/OOS split. Rebuild H1 path to 1080h (45d), then test trailing 12..25%.
Reports MFE buckets >=10/20/30/50%, signal rate, capture and BUY->TOP->SELL timing.
"""
exec(open('sar_phase13b_exit_fast.py').read().split("# 90+ meaningful configurations")[0])

# EV from Phase13B. Extend each event path from its asset H1 dataframe already loaded in D.
# D is asset -> dataframe, with timestamp/index and OHLC columns used by Phase13B.
def _col(df,names):
 for x in names:
  if x in df.columns:return x
 raise KeyError(str(names))

for d in EV:
 sym=d.get('s',d.get('sym',d.get('asset')))
 df=D[sym]
 tc=_col(df,['ts','time','timestamp','open_time']); cc=_col(df,['close','cl','c']); hc=_col(df,['high','hi','h']); lc=_col(df,['low','lo','l'])
 t0=pd.Timestamp(d['ix'])
 tt=pd.to_datetime(df[tc],unit='ms',errors='coerce') if np.issubdtype(df[tc].dtype,np.number) else pd.to_datetime(df[tc])
 ix=np.where(tt.values>=np.datetime64(t0))[0]
 if len(ix):
  a=ix[0];b=min(len(df),a+1081);d['ts17']=tt.iloc[a:b].to_numpy();d['cl17']=df[cc].iloc[a:b].astype(float).to_numpy();d['hi17']=df[hc].iloc[a:b].astype(float).to_numpy();d['lo17']=df[lc].iloc[a:b].astype(float).to_numpy()
 else:d['ts17']=d['ts'];d['cl17']=d['cl'];d['hi17']=d['hi'];d['lo17']=d['lo']
 d['mfe17']=max(0,float(np.max(d['hi17'])/d['e']-1))

et=np.array([pd.Timestamp(d['ix']).value for d in EV]);order=np.argsort(et);cuts=np.linspace(0,len(order),6,dtype=int);base=order[cuts[1]:]
print('EVENTS17B|n=%d|oos=%d|horizon=1080h'%(len(EV),len(base)),flush=True)

def one(d,q):
 e=d['e'];h=d['hi17'];c=d['cl17'];n=min(len(c),1081);pk=h[0];pkj=0;sell=None
 for j in range(1,n):
  if h[j]>pk:pk=h[j];pkj=j
  if c[j]<=pk*(1-q):sell=j;break
 # Eventual top over full 45d is used only for ex-post evaluation/timing, never for signal.
 topj=int(np.argmax(h[:n]));top=float(h[topj]);mfe=top/e-1
 if sell is None:return mfe,topj/24.,np.nan,np.nan,np.nan,np.nan,0
 px=float(c[sell]);ret=px/e-1-.003;cap=ret/mfe if mfe>0 else np.nan
 lag=(sell-topj)/24. if sell>=topj else -(topj-sell)/24.
 give=(top-px)/top
 return mfe,topj/24.,sell/24.,lag,ret,cap,1

for th in [.10,.20,.30,.50]:
 ids=[i for i in base if EV[i]['mfe17']>=th]
 print('BUCKET17B|mfe>=%.2f|n=%d'%(th,len(ids)),flush=True)
 for pct in range(12,26):
  q=pct/100.;a=np.asarray([one(EV[i],q) for i in ids],dtype=float);sig=a[:,6]>0
  topmed=np.nanmedian(a[:,1]);sr=sig.mean()
  if sig.any():
   sellmed=np.nanmedian(a[sig,2]);lagmed=np.nanmedian(a[sig,3]);lagavg=np.nanmean(a[sig,3]);ret=np.nanmean(a[sig,4]);cap=np.nanmedian(a[sig,5]);win=np.mean(a[sig,4]>0)
   # fraction whose SELL occurs after eventual 45d top; negative lag means premature sell
   post=np.mean(a[sig,3]>=0);prem=np.mean(a[sig,3]<0)
  else:sellmed=lagmed=lagavg=ret=cap=win=post=prem=np.nan
  print('TRAIL17B|mfe%.2f|trail%d|n%d|signal%.3f|top_day_med%.2f|sell_day_med%.2f|top_to_sell_med%.2f|lagavg%.2f|mean%.4f|win%.3f|cap%.3f|posttop%.3f|premature%.3f'%(th,pct,len(ids),sr,topmed,sellmed,lagmed,lagavg,ret,win,cap,post,prem),flush=True)
print('DONE17B',flush=True)
