"""Phase17B v2: isolated BUY->TOP->SELL trailing study, 45d paths, no Phase17/13B tournament execution."""
# Bootstrap only Phase8 data construction. Extend its future H1 path from 16d to 46d before executing.
src=open('sar_phase8_presar_intraday.py').read()
src=src.replace("day+pd.Timedelta(days=16)","day+pd.Timedelta(days=46)")
src=src.split("for th in [.45,.50,.55,.60]:")[0]
exec(src)
print('BOOT17B2|phase8_data_only',flush=True)

EV=[]
for ix,r in t.iterrows():
 p=r['_path'].iloc[:1081].copy();e=float(r.entry)
 if len(p)<2:continue
 EV.append({'ix':ix,'e':e,'ts':p.index,'hi':p.high.to_numpy(float),'lo':p.low.to_numpy(float),'cl':p.close.to_numpy(float)})
N=len(EV);order=np.argsort([pd.Timestamp(d['ix']).value for d in EV]);cuts=np.linspace(0,N,6,dtype=int);base=order[cuts[1]:]
print('EVENTS17B|n=%d|oos=%d|horizon=1080h'%(N,len(base)),flush=True)

def one(d,q):
 e=d['e'];h=d['hi'];c=d['cl'];n=len(c);pk=h[0];sell=None
 for j in range(1,n):
  pk=max(pk,h[j])
  if c[j]<=pk*(1-q):sell=j;break
 topj=int(np.argmax(h));top=float(h[topj]);mfe=top/e-1
 if sell is None:return mfe,topj/24.,np.nan,np.nan,np.nan,np.nan,0
 px=float(c[sell]);ret=px/e-1-.003;cap=ret/mfe if mfe>0 else np.nan;lag=(sell-topj)/24.
 return mfe,topj/24.,sell/24.,lag,ret,cap,1

for th in [.10,.20,.30,.50]:
 ids=[i for i in base if max(EV[i]['hi'])/EV[i]['e']-1>=th]
 print('BUCKET17B|mfe>=%.2f|n=%d'%(th,len(ids)),flush=True)
 for pct in range(12,26):
  a=np.asarray([one(EV[i],pct/100.) for i in ids],float);sig=a[:,6]>0;sr=sig.mean();topmed=np.nanmedian(a[:,1])
  if sig.any():
   sellmed=np.nanmedian(a[sig,2]);lagmed=np.nanmedian(a[sig,3]);lagavg=np.nanmean(a[sig,3]);ret=np.nanmean(a[sig,4]);cap=np.nanmedian(a[sig,5]);win=np.mean(a[sig,4]>0);post=np.mean(a[sig,3]>=0);prem=np.mean(a[sig,3]<0)
  else:sellmed=lagmed=lagavg=ret=cap=win=post=prem=np.nan
  print('TRAIL17B|mfe%.2f|trail%d|n%d|signal%.3f|top_day_med%.2f|sell_day_med%.2f|top_to_sell_med%.2f|lagavg%.2f|mean%.4f|win%.3f|cap%.3f|posttop%.3f|premature%.3f'%(th,pct,len(ids),sr,topmed,sellmed,lagmed,lagavg,ret,win,cap,post,prem),flush=True)
print('DONE17B',flush=True)
