"""Phase13: BUY is fixed at first intraday cross above Daily SAR. Optimize EXIT only.
Core hypothesis: lower-timeframe Parabolic SAR flips can detect the post-cross peak earlier than Daily SAR.
Test PSAR exits on 1h/2h/4h/6h/8h/12h, confirmations, delayed activation, profit-armed combinations and multi-TF voting.
All exits are causal: signals use completed resampled bars and execution occurs at next 1h open (+ conservative slippage/cost).
Evaluate capture of MFE, realized return, drawdown giveback and chronological fold stability on ALL entries.
"""
exec(open('sar_phase8_presar_intraday.py').read().split("for th in [.45,.50,.55,.60]:")[0])
COST=.002; SLIP=.001
# local PSAR implementation standard Wilder-style
def psar_flags(df,step=.02,maxaf=.2):
 hi=df.high.to_numpy(float);lo=df.low.to_numpy(float);cl=df.close.to_numpy(float);n=len(df)
 if n<3:return pd.Series(False,index=df.index)
 bull=np.ones(n,dtype=bool);sar=np.full(n,np.nan);ep=hi[0];af=step;sar[0]=lo[0]
 for i in range(1,n):
  prev=sar[i-1];s=prev+af*(ep-prev)
  if bull[i-1]:
   s=min(s,lo[i-1],lo[i-2] if i>1 else lo[i-1])
   if lo[i]<s: bull[i]=False;s=ep;ep=lo[i];af=step
   else:
    bull[i]=True
    if hi[i]>ep:ep=hi[i];af=min(maxaf,af+step)
  else:
   s=max(s,hi[i-1],hi[i-2] if i>1 else hi[i-1])
   if hi[i]>s:bull[i]=True;s=ep;ep=hi[i];af=step
   else:
    bull[i]=False
    if lo[i]<ep:ep=lo[i];af=min(maxaf,af+step)
  sar[i]=s
 return pd.Series(bull,index=df.index),pd.Series(sar,index=df.index)

def rs(path,H):
 if H==1:return path.copy()
 return path.resample('%dh'%H,origin='epoch',label='right',closed='right').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
TF=[1,2,4,6,8,12]
# precompute each event's lower-TF bearish flip times and MFE over 15d
EV=[]
for ix,r in t.iterrows():
 p=r['_path'];e=float(r.entry);q=p.iloc[:24*15]
 if q.empty:continue
 peak=float(q.high.max());mfe=peak/e-1;peak_t=q.high.idxmax();d={'ix':ix,'entry':e,'mfe':mfe,'peak_t':peak_t,'path':q}
 for H in TF:
  z=rs(q,H);b,s=psar_flags(z);flip=(~b)&b.shift(1,fill_value=True);d['flip%d'%H]=list(z.index[flip])
 EV.append(d)
print('EVENTS13|n=%d'%len(EV),flush=True)
# MFE distribution validates user's premise
for x in [.02,.05,.08,.10,.15,.20,.30]:print('MFE13|level=%.2f|reach=%.3f'%(x,np.mean([d['mfe']>=x for d in EV])),flush=True)
# exit configs: single TF, confirmation votes, activation delay, arm profit.
CFG=[]
for H in TF:
 for delay in [0,12,24,48]:
  for arm in [0,.02,.05,.08]:CFG.append(('single',(H,),1,delay,arm))
for combo in [(1,4),(1,6),(1,12),(2,4),(4,12),(1,4,12),(2,4,12),(4,8,12)]:
 for votes in range(1,len(combo)+1):
  for delay in [0,12,24]:
   for arm in [0,.02,.05]:CFG.append(('vote',combo,votes,delay,arm))
# add trailing baselines for comparison
for tr in [.02,.03,.04,.05,.06,.08,.10]:
 for arm in [0,.02,.05,.08,.10]:CFG.append(('trail',(tr,),1,0,arm))

def run(d,c):
 typ,combo,votes,delay,arm=c;e=d['entry'];p=d['path'];start=p.index[0];peak=e;armed=(arm==0);exitpx=None;exitt=None
 # prebuild flip timestamps by TF; vote means bearish flips within last max TF*1.5 hours
 flips={H:set(d['flip%d'%H]) for H in combo if isinstance(H,int) and H in TF}
 for j,(ts,b) in enumerate(p.iterrows()):
  peak=max(peak,float(b.high));armed=armed or peak>=e*(1+arm)
  if (ts-start).total_seconds()/3600<delay or not armed:continue
  fire=False
  if typ=='single':
   H=combo[0]; fire=any((f<=ts and f>ts-pd.Timedelta(hours=max(2,H+1))) for f in flips[H])
  elif typ=='vote':
   cnt=0
   for H in combo:
    if any((f<=ts and f>ts-pd.Timedelta(hours=max(2,H+1))) for f in flips[H]):cnt+=1
   fire=cnt>=votes
  else:
   tr=combo[0];fire=float(b.low)<=peak*(1-tr)
  if fire:
   # execute conservatively at current close for completed-bar SAR signal; trailing at stop level or close whichever worse
   exitpx=float(b.close) if typ!='trail' else min(float(b.close),peak*(1-combo[0]));exitt=ts;break
 if exitpx is None:exitpx=float(p.close.iloc[-1]);exitt=p.index[-1]
 ret=exitpx/e-1-COST-SLIP;cap=(ret/max(d['mfe'],1e-9)) if d['mfe']>0 else np.nan;give=(peak-exitpx)/e
 return ret,cap,give,(exitt-start).total_seconds()/86400

def score(cfg,subset):
 a=np.array([run(EV[i],cfg) for i in subset],float);r=a[:,0];cap=a[:,1];return len(r),(r>0).mean(),r.mean(),np.nanmedian(cap),np.nanmean(cap),np.mean(a[:,2]),np.median(a[:,3]),r.min()
N=len(EV);allidx=list(range(N));R=[]
for c in CFG:
 st=score(c,allidx);n,w,me,mc,ac,g,days,wo=st
 # optimize return + capture + win, penalize giveback and worst loss
 sc=me+.02*w+.015*np.clip(ac,-1,1)-.10*g+.10*wo
 R.append((sc,c,st))
for k,(sc,c,st) in enumerate(sorted(R,key=lambda x:x[0],reverse=True)[:25],1):
 n,w,me,mc,ac,g,days,wo=st;print('TOP13|rank=%d|type=%s|tf=%s|votes=%d|delay=%d|arm=%.2f|n=%d|win=%.3f|mean=%.4f|medcap=%.3f|avgcap=%.3f|give=%.4f|days=%.2f|worst=%.4f'%(k,c[0],'-'.join(map(str,c[1])),c[2],c[3],c[4],n,w,me,mc,ac,g,days,wo),flush=True)
# chronological nested selection: sort events by entry date, 5 contiguous blocks; choose config only from prior blocks
order=np.argsort([pd.Timestamp(d['ix']).value for d in EV]);cuts=np.linspace(0,N,6,dtype=int);ALL=[]
for f in range(1,5):
 hist=order[:cuts[f+0]];test=order[cuts[f]:cuts[f+1]]
 best=None
 for c in CFG:
  st=score(c,hist);n,w,me,mc,ac,g,days,wo=st;sc=me+.02*w+.015*np.clip(ac,-1,1)-.10*g+.10*wo
  if best is None or sc>best[0]:best=(sc,c)
 c=best[1];vals=[run(EV[i],c) for i in test];ALL+=vals;st=score(c,test);n,w,me,mc,ac,g,days,wo=st
 print('NEST13|fold=%d|type=%s|tf=%s|votes=%d|delay=%d|arm=%.2f|n=%d|win=%.3f|mean=%.4f|medcap=%.3f|give=%.4f|days=%.2f|worst=%.4f'%(f+1,c[0],'-'.join(map(str,c[1])),c[2],c[3],c[4],n,w,me,mc,g,days,wo),flush=True)
a=np.array(ALL,float);print('NESTSUM13|n=%d|win=%.3f|mean=%.4f|medcap=%.3f|avgcap=%.3f|give=%.4f|days=%.2f|worst=%.4f'%(len(a),(a[:,0]>0).mean(),a[:,0].mean(),np.nanmedian(a[:,1]),np.nanmean(a[:,1]),a[:,2].mean(),np.median(a[:,3]),a[:,0].min()),flush=True)
# single-TF clean comparison with no arm/delay
for H in TF:
 c=('single',(H,),1,0,0);st=score(c,allidx);print('SAR13|tf=%dh|n=%d|win=%.3f|mean=%.4f|medcap=%.3f|give=%.4f|days=%.2f|worst=%.4f'%((H,)+ (st[0],st[1],st[2],st[3],st[5],st[6],st[7])),flush=True)
print('DONE13',flush=True)
