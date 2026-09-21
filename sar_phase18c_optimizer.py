"""Phase18C: causal combinatorial SAR exit optimizer on exact Phase17C OOS.
Discovery first 65% chronological OOS, untouched validation last 35%.
Enumerates nth bearish flips, cross-TF confirmations and simultaneous bearish states.
TOP is evaluation-only. Objective prioritizes avoiding >5/7.5/10/15% missed upside,
then giveback/return, with coverage and complexity penalties.
"""
exec(open('sar_phase17c_cycle_integrity.py').read().split("EV=sorted(EV,key=lambda x:x['ts']);cut=int(len(EV)*.20);O=EV[cut:]")[0])
EV=sorted(EV,key=lambda x:x['ts']);O=EV[int(len(EV)*.20):]
N=len(O);split=int(N*.65);print('OOS18C|events=%d|oos=%d|train=%d|test=%d'%(len(EV),N,split,N-split),flush=True)
TF=[1,2,4,6,8,12]
def psar_state(df,step=.02,maxaf=.2):
 hi=df.high.to_numpy(float);lo=df.low.to_numpy(float);n=len(df);bull=np.ones(n,dtype=bool);sar=np.full(n,np.nan)
 if not n:return bull,sar
 ep=hi[0];af=step;sar[0]=lo[0]
 for i in range(1,n):
  s=sar[i-1]+af*(ep-sar[i-1])
  if bull[i-1]:
   s=min(s,lo[i-1],lo[i-2] if i>1 else lo[i-1])
   if lo[i]<s:bull[i]=False;s=ep;ep=lo[i];af=step
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
 return bull,sar
def bars(p,tf):
 if tf==1:return p
 return p.resample('%dh'%tf,origin='epoch',label='right',closed='right').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
def first_after(arr,t0,t1=None):
 q=arr[arr>=t0];q=q if t1 is None else q[q<=t1];return q[0] if len(q) else None
# Candidate rule descriptors: (name, type, params, complexity)
rules=[]
for tf in TF:
 for nth in [1,2,3]:rules.append(('%dH_F%d'%(tf,nth),'nth',(tf,nth),1+nth*.15))
# nth primary flip followed by confirmation TF within lag
for a in [2,4,6,8]:
 for nth in [1,2,3]:
  for b in TF:
   if b<=a:continue
   for lag in [12,24,48,72]:rules.append(('%dH_F%d__%dH_%dh'%(a,nth,b,lag),'confirm',(a,nth,b,lag),2.0+nth*.15+lag/240.))
# simultaneous bearish sets, checked causally each hour; require primary 4H plus others
sets=[(4,6),(4,8),(4,12),(4,6,8),(4,6,12),(4,8,12),(4,6,8,12),(2,4,6),(2,4,6,8),(1,2,4,6)]
for s in sets:
 for minage in [0,4,8,12]:rules.append(('SIM_'+'_'.join(map(str,s))+'__A%d'%minage,'sim',(s,minage),len(s)+minage/24.))
print('RULES18C|%d'%len(rules),flush=True)
H={}
for sym in sorted(set(x['sym'] for x in O)):
 xs=[x for x in O if x['sym']==sym];a=min(x['ts'] for x in xs)-pd.Timedelta(days=4);e=max(x['ts'] for x in xs)+pd.Timedelta(hours=1081)
 try:H[sym]=hourly(sym,int(a.timestamp()*1000),int(e.timestamp()*1000));print('H18C|%s|%d'%(sym,len(H[sym])),flush=True)
 except Exception as ex:H[sym]=pd.DataFrame();print('FAIL18C|%s|%s'%(sym,str(ex)[:100]),flush=True)
out=[]
for ci,x in enumerate(O):
 p=H.get(x['sym'],pd.DataFrame());p=p[(p.index>=x['ts'])&(p.index<x['ts']+pd.Timedelta(hours=1081))]
 if p.empty:continue
 dend=x['sell'] if x['sell'] is not None else p.index[-1];cyc=p[p.index<=dend]
 if cyc.empty:continue
 top=float(cyc.high.max());Z={};S={};F={}
 for tf in TF:
  z=bars(p,tf);b,_=psar_state(z);fi=np.where((~b)&np.r_[True,b[:-1]])[0];Z[tf]=z;S[tf]=b;F[tf]=z.index[fi]
 def emit(name,ts,complexity):
  if ts is None or ts>dend:return
  ix=p.index.searchsorted(ts,'left')
  if ix>=len(p):return
  px=float(p.close.iloc[ix]);remain=max(0.,top/px-1.);give=max(0.,1-px/top);ret=px/float(x['entry'])-1
  out.append((ci,name,remain,give,ret,(dend-p.index[ix]).total_seconds()/3600,complexity))
 for name,typ,pa,cx in rules:
  ts=None
  if typ=='nth':
   tf,nth=pa
   if len(F[tf])>=nth:ts=F[tf][nth-1]
  elif typ=='confirm':
   a,nth,b,lag=pa
   if len(F[a])>=nth:
    fa=F[a][nth-1];ts=first_after(F[b],fa,fa+pd.Timedelta(hours=lag))
  else:
   ss,minage=pa
   # first hourly timestamp when all requested TF states are bearish; minage since BUY prevents trivial immediate exits
   for tt in p.index[p.index>=x['ts']+pd.Timedelta(hours=minage)]:
    ok=True
    for tf in ss:
     j=Z[tf].index.searchsorted(tt,'right')-1
     if j<0 or S[tf][j]:ok=False;break
    if ok:ts=tt;break
  emit(name,ts,cx)
R=pd.DataFrame(out,columns=['ci','rule','remain','give','ret','daily_lead','complexity']);print('ROWS18C|%d'%len(R),flush=True)
def metrics(q,den):
 return {'n':len(q),'cov':len(q)/den,'ret':np.median(q.ret) if len(q) else np.nan,'give':np.median(q.give) if len(q) else np.nan,'rem':np.median(q.remain) if len(q) else np.nan,'g5':np.mean(q.remain>.05) if len(q) else 1.,'g75':np.mean(q.remain>.075) if len(q) else 1.,'g10':np.mean(q.remain>.10) if len(q) else 1.,'g15':np.mean(q.remain>.15) if len(q) else 1.}
def score(m,cx):
 # lower is better; strongly penalize low coverage and large missed upside, mildly complexity
 return 3*m['g5']+2*m['g75']+1.5*m['g10']+m['g15']+1.5*(1-m['cov'])+.35*m['give']-.15*m['ret']+.015*cx
train_ids=set(range(split));test_ids=set(range(split,N));rank=[]
for name,typ,pa,cx in rules:
 q=R[(R.rule==name)&R.ci.isin(train_ids)];m=metrics(q,split);rank.append((score(m,cx),name,cx,m))
rank.sort()
for k,(sc,name,cx,m) in enumerate(rank[:20],1):print('TRAIN18C|%d|%s|score=%.4f|n=%d|cov=%.3f|ret=%.4f|give=%.4f|rem=%.4f|gt5=%.3f|gt7.5=%.3f|gt10=%.3f|gt15=%.3f'%(k,name,sc,m['n'],m['cov'],m['ret'],m['give'],m['rem'],m['g5'],m['g75'],m['g10'],m['g15']),flush=True)
# untouched validation only for top-20 selected on train
for k,(sc,name,cx,mt) in enumerate(rank[:20],1):
 q=R[(R.rule==name)&R.ci.isin(test_ids)];m=metrics(q,N-split)
 print('TEST18C|%d|%s|n=%d|cov=%.3f|ret=%.4f|give=%.4f|rem=%.4f|gt5=%.3f|gt7.5=%.3f|gt10=%.3f|gt15=%.3f|daily_lead=%.1f'%(k,name,m['n'],m['cov'],m['ret'],m['give'],m['rem'],m['g5'],m['g75'],m['g10'],m['g15'],np.nanmedian(q.daily_lead) if len(q) else np.nan),flush=True)
print('DONE18C',flush=True)