"""Phase17: 45-day professional exit tournament with explicit exit timing/censoring."""
# Load event source only; rebuild paths to 1080H instead of Phase13B's 360H cap.
exec(open('sar_phase8_presar_intraday.py').read().split("for th in [.45,.50,.55,.60]:")[0])
COST=.002;SLIP=.001;TF=[1,2,4,6,8,12];HORIZON=1080

def psar(df,step=.02,maxaf=.2):
 hi=df.high.to_numpy(float);lo=df.low.to_numpy(float);n=len(df);bull=np.ones(n,dtype=bool);sar=np.full(n,np.nan);ep=hi[0];af=step;sar[0]=lo[0]
 for i in range(1,n):
  s=sar[i-1]+af*(ep-sar[i-1])
  if bull[i-1]:
   s=min(s,lo[i-1],lo[i-2] if i>1 else lo[i-1])
   if lo[i]<s:bull[i]=False;s=ep;ep=lo[i];af=step
   elif hi[i]>ep:ep=hi[i];af=min(maxaf,af+step)
  else:
   s=max(s,hi[i-1],hi[i-2] if i>1 else hi[i-1])
   if hi[i]>s:bull[i]=True;s=ep;ep=hi[i];af=step
   else:
    bull[i]=False
    if lo[i]<ep:ep=lo[i];af=min(maxaf,af+step)
  sar[i]=s
 return bull

def ema(x,n):
 a=2/(n+1);o=np.empty(len(x));o[0]=x[0]
 for i in range(1,len(x)):o[i]=a*x[i]+(1-a)*o[i-1]
 return o

def atr(h,l,c,n):
 tr=np.empty(len(c));tr[0]=h[0]-l[0]
 for i in range(1,len(c)):tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
 o=np.empty(len(c));o[0]=tr[0];a=1/n
 for i in range(1,len(c)):o[i]=(1-a)*o[i-1]+a*tr[i]
 return o

EV=[]
for ix,r in t.iterrows():
 p=r['_path'].iloc[:HORIZON].copy();e=float(r.entry)
 if len(p)<48:continue
 d={'ix':ix,'e':e,'ts':p.index,'hi':p.high.to_numpy(float),'lo':p.low.to_numpy(float),'cl':p.close.to_numpy(float)};d['mfe']=d['hi'].max()/e-1
 for H in TF:
  z=p if H==1 else p.resample('%dh'%H,origin='epoch',label='right',closed='right').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna();b=psar(z);flip=np.where((~b)&np.r_[True,b[:-1]])[0];d['f%d'%H]=z.index[flip]
 EV.append(d)
print('EVENTS17|n=%d|horizon=%dh'%(len(EV),HORIZON),flush=True)

def exit_idx(d,r):
 typ,p=r;c=d['cl'];h=d['hi'];l=d['lo'];ts=d['ts'];n=len(c)
 if typ=='SAR':
  fs=d['f%d'%p]
  for j in range(1,n):
   if any((f<=ts[j]) and (f>ts[j-1]) for f in fs):return j,True
 elif typ=='TRAIL':
  pk=h[0]
  for j in range(1,n):
   pk=max(pk,h[j])
   if c[j]<=pk*(1-p):return j,True
 elif typ=='CHAND':
  per,k=p;A=atr(h,l,c,per);pk=h[0]
  for j in range(max(2,per),n):
   pk=max(pk,h[j])
   if c[j]<=pk-k*A[j]:return j,True
 elif typ=='LOW':
  for j in range(p,n):
   if c[j]<np.min(l[j-p:j]):return j,True
 elif typ=='EMA':
  E=ema(c,p)
  for j in range(max(2,p),n):
   if c[j]<E[j]:return j,True
 return n-1,False

RULES=[]
for H in TF:RULES.append(('SAR',H))
for q in [.08,.10,.12,.15,.18,.20]:RULES.append(('TRAIL',q))
for per in [48,72]:
 for k in [3.,4.,5.,6.]:RULES.append(('CHAND',(per,k)))
for w in [48,72,96,120,144,168,240]:RULES.append(('LOW',w))
for w in [72,96,120,144,168,240]:RULES.append(('EMA',w))
et=np.array([pd.Timestamp(d['ix']).value for d in EV]);order=np.argsort(et);cuts=np.linspace(0,len(order),6,dtype=int);base=order[cuts[1]:]
print('TOURN17|rules=%d|oos=%d'%(len(RULES),len(base)),flush=True)
for threshold in [.10,.20,.30]:
 ids=[i for i in base if EV[i]['mfe']>=threshold];R=[];print('BUCKET17|mfe>=%.2f|n=%d'%(threshold,len(ids)),flush=True)
 for r in RULES:
  z=[]
  for i in ids:
   d=EV[i];j,signal=exit_idx(d,r);px=d['cl'][j];ret=px/d['e']-1-COST-SLIP;cap=ret/d['mfe'];z.append((ret,cap,j/24.,float(signal)))
  a=np.asarray(z);sig=a[:,3]>0;sd=a[sig,2] if sig.any() else np.array([np.nan]);
  dist=[np.mean(sig & (a[:,2]<=x)) for x in [3,5,7,10,15,20,30]]
  s={'mean':a[:,0].mean(),'win':(a[:,0]>0).mean(),'cap':np.median(a[:,1]),'signal':sig.mean(),'med':np.nanmedian(sd),'avg':np.nanmean(sd),'p25':np.nanpercentile(sd,25),'p75':np.nanpercentile(sd,75),'dist':dist,'worst':a[:,0].min()}
  score=s['cap']+.25*s['mean']-.10*(1-s['signal']);R.append((score,r,s))
 for rank,(sc,r,s) in enumerate(sorted(R,key=lambda x:x[0],reverse=True)[:15],1):
  print('TOP17|mfe%.2f|%d|%s|%s|mean%.4f|win%.3f|cap%.3f|signal%.3f|days_med%.2f|days_avg%.2f|p25%.2f|p75%.2f|d3/5/7/10/15/20/30=%s|worst%.4f'%(threshold,rank,r[0],str(r[1]),s['mean'],s['win'],s['cap'],s['signal'],s['med'],s['avg'],s['p25'],s['p75'],'/'.join('%.3f'%x for x in s['dist']),s['worst']),flush=True)
print('DONE17',flush=True)
