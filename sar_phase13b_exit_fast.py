"""Phase13B fast exit-only study. BUY fixed at intraday Daily-SAR cross. No entry filtering.
Tests dozens of causal lower-TF SAR exit rules on all 1325 events, optimized for Railway runtime.
"""
exec(open('sar_phase8_presar_intraday.py').read().split("for th in [.45,.50,.55,.60]:")[0])
COST=.002; SLIP=.001; TF=[1,2,4,6,8,12]
def psar(df,step=.02,maxaf=.2):
 hi=df.high.to_numpy(float);lo=df.low.to_numpy(float);n=len(df);bull=np.ones(n,dtype=bool);sar=np.full(n,np.nan);ep=hi[0];af=step;sar[0]=lo[0]
 for i in range(1,n):
  s=sar[i-1]+af*(ep-sar[i-1])
  if bull[i-1]:
   s=min(s,lo[i-1],lo[i-2] if i>1 else lo[i-1])
   if lo[i]<s:bull[i]=False;s=ep;ep=lo[i];af=step
   else:
    if hi[i]>ep:ep=hi[i];af=min(maxaf,af+step)
  else:
   s=max(s,hi[i-1],hi[i-2] if i>1 else hi[i-1])
   if hi[i]>s:bull[i]=True;s=ep;ep=hi[i];af=step
   else:
    bull[i]=False
    if lo[i]<ep:ep=lo[i];af=min(maxaf,af+step)
  sar[i]=s
 return bull
EV=[]
for ix,r in t.iterrows():
 p=r['_path'].iloc[:360].copy();e=float(r.entry)
 if len(p)<2:continue
 d={'ix':ix,'e':e,'ts':p.index,'hi':p.high.to_numpy(float),'lo':p.low.to_numpy(float),'cl':p.close.to_numpy(float)};d['mfe']=d['hi'].max()/e-1
 for H in TF:
  z=p if H==1 else p.resample('%dh'%H,origin='epoch',label='right',closed='right').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
  b=psar(z);flip=np.where((~b)&np.r_[True,b[:-1]])[0];d['f%d'%H]=z.index[flip]
 EV.append(d)
print('EVENTS13B|n=%d'%len(EV),flush=True)
for x in [.02,.05,.08,.10,.15,.20,.30]:print('MFE13B|%.2f|%.3f'%(x,np.mean([d['mfe']>=x for d in EV])),flush=True)
# 90+ meaningful configurations
CFG=[]
for H in TF:
 for delay in [0,12,24]:
  for arm in [0,.02,.05,.08]:CFG.append(('S',(H,),1,delay,arm))
for combo in [(1,4),(1,12),(2,4),(4,12),(1,4,12),(2,4,12),(4,8,12)]:
 for votes in range(1,len(combo)+1):
  for arm in [0,.02,.05]:CFG.append(('V',combo,votes,0,arm))
for tr in [.02,.03,.04,.05,.06,.08,.10]:
 for arm in [0,.02,.05,.08]:CFG.append(('T',(tr,),1,0,arm))
# fast numpy execution; lower-TF SAR signal only from completed bars
def run(d,c):
 typ,combo,votes,delay,arm=c;e=d['e'];hi=d['hi'];lo=d['lo'];cl=d['cl'];ts=d['ts'];peak=e;armed=arm==0
 # boolean fire arrays precomputed by timestamp membership windows
 fire=np.zeros(len(ts),dtype=bool)
 if typ in ('S','V'):
  cnt=np.zeros(len(ts),dtype=np.int8)
  for H in combo:
   ff=d['f%d'%H]
   # completed flip stays actionable H+1 hours
   for f in ff:
    cnt += np.asarray((ts>=f)&(ts<f+pd.Timedelta(hours=max(2,H+1))),dtype=np.int8)
  fire=cnt>=votes
 for j in range(len(ts)):
  peak=max(peak,hi[j]);armed=armed or peak>=e*(1+arm)
  if j<delay or not armed:continue
  hit=fire[j] if typ!='T' else lo[j]<=peak*(1-combo[0])
  if hit:
   px=cl[j] if typ!='T' else min(cl[j],peak*(1-combo[0]));ret=px/e-1-COST-SLIP
   return ret,ret/max(d['mfe'],1e-9),max(0,(peak-px)/e),j/24
 px=cl[-1];ret=px/e-1-COST-SLIP;return ret,ret/max(d['mfe'],1e-9),max(0,(peak-px)/e),(len(ts)-1)/24
def stat(c,ids):
 a=np.array([run(EV[i],c) for i in ids]);r=a[:,0];return (len(r),(r>0).mean(),r.mean(),np.nanmedian(a[:,1]),np.nanmean(a[:,1]),a[:,2].mean(),np.median(a[:,3]),r.min())
def obj(s):
 n,w,me,mc,ac,g,days,wo=s;return me+.02*w+.012*np.clip(ac,-1,1)-.08*g+.08*wo
N=len(EV);ids=np.arange(N);R=[]
for k,c in enumerate(CFG):
 s=stat(c,ids);R.append((obj(s),c,s))
 if (k+1)%25==0:print('PROG13B|%d/%d'%(k+1,len(CFG)),flush=True)
for k,(o,c,s) in enumerate(sorted(R,reverse=True,key=lambda x:x[0])[:20],1):
 n,w,me,mc,ac,g,days,wo=s;print('TOP13B|%d|%s|%s|v%d|d%d|a%.2f|n%d|win%.3f|mean%.4f|medcap%.3f|give%.4f|days%.2f|worst%.4f'%(k,c[0],'-'.join(map(str,c[1])),c[2],c[3],c[4],n,w,me,mc,g,days,wo),flush=True)
# chronological nested selection: first 20% warmup then 4 forward blocks
order=np.argsort([pd.Timestamp(d['ix']).value for d in EV]);cuts=np.linspace(0,N,6,dtype=int);ALL=[]
for f in range(1,5):
 hist=order[:cuts[f]];test=order[cuts[f]:cuts[f+1]];best=max((obj(stat(c,hist)),c) for c in CFG);c=best[1];vals=[run(EV[i],c) for i in test];ALL+=vals;s=stat(c,test)
 print('NEST13B|f%d|%s|%s|v%d|d%d|a%.2f|n%d|win%.3f|mean%.4f|medcap%.3f|give%.4f|days%.2f|worst%.4f'%(f+1,c[0],'-'.join(map(str,c[1])),c[2],c[3],c[4],s[0],s[1],s[2],s[3],s[5],s[6],s[7]),flush=True)
a=np.array(ALL);print('NESTSUM13B|n=%d|win=%.3f|mean=%.4f|medcap=%.3f|avgcap=%.3f|give=%.4f|days=%.2f|worst=%.4f'%(len(a),(a[:,0]>0).mean(),a[:,0].mean(),np.nanmedian(a[:,1]),np.nanmean(a[:,1]),a[:,2].mean(),np.median(a[:,3]),a[:,0].min()),flush=True)
for H in TF:
 s=stat(('S',(H,),1,0,0),ids);print('SAR13B|%dh|win=%.3f|mean=%.4f|medcap=%.3f|give=%.4f|days=%.2f|worst=%.4f'%(H,s[1],s[2],s[3],s[5],s[6],s[7]),flush=True)
print('DONE13B',flush=True)
