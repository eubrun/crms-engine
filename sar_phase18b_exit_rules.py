"""Phase 18B — causal SAR exit rule comparison on exact Phase17C OOS universe.
Primary objective: preserve large gains, not the final 2%. Compare 4H bearish exit with
4H+6H confirmation, 4H persistence, and 1H->2H->4H cascade. Report missed upside >5/7.5/10/15%.
"""
exec(open('sar_phase17c_cycle_integrity.py').read().split("EV=sorted(EV,key=lambda x:x['ts']);cut=int(len(EV)*.20);O=EV[cut:]")[0])
EV=sorted(EV,key=lambda x:x['ts']);O=EV[int(len(EV)*.20):]
print('OOS18B|events=%d|oos=%d'%(len(EV),len(O)),flush=True)

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

def flips(z):
 b,_=psar_state(z);bear=z.index[np.where((~b)&np.r_[True,b[:-1]])[0]];return b,bear

H={}
for sym in sorted(set(x['sym'] for x in O)):
 xs=[x for x in O if x['sym']==sym];a=min(x['ts'] for x in xs)-pd.Timedelta(days=4);e=max(x['ts'] for x in xs)+pd.Timedelta(hours=1081)
 try:H[sym]=hourly(sym,int(a.timestamp()*1000),int(e.timestamp()*1000));print('H18B|%s|%d'%(sym,len(H[sym])),flush=True)
 except Exception as ex:H[sym]=pd.DataFrame();print('FAIL18B|%s|%s'%(sym,str(ex)[:120]),flush=True)

rules=['4H','4H_PERSIST8','4H_6H','CASCADE124']
out=[]
for ci,x in enumerate(O):
 p=H.get(x['sym'],pd.DataFrame());p=p[(p.index>=x['ts'])&(p.index<x['ts']+pd.Timedelta(hours=1081))]
 if p.empty:continue
 dend=x['sell'] if x['sell'] is not None else p.index[-1];cyc=p[p.index<=dend]
 if cyc.empty:continue
 top=float(cyc.high.max());tf={k:bars(p,k) for k in [1,2,4,6]};st={};bf={}
 for k,z in tf.items():st[k],bf[k]=flips(z)
 candidates={r:None for r in rules}
 # first 4H bearish flip after BUY
 if len(bf[4]):
  f4=bf[4][0];candidates['4H']=f4
  # persistence: still bearish 8 clock-hours after the 4H flip
  chk=f4+pd.Timedelta(hours=8);pos=tf[4].index.searchsorted(chk,side='right')-1
  if pos>=0 and pos<len(st[4]) and not st[4][pos]:candidates['4H_PERSIST8']=tf[4].index[pos]
  # first 6H bearish flip after the 4H flip, capped at 24h confirmation lag
  q6=bf[6][(bf[6]>=f4)&(bf[6]<=f4+pd.Timedelta(hours=24))]
  if len(q6):candidates['4H_6H']=q6[0]
 # causal cascade: 1H bearish then 2H within 24h then 4H within 24h of 2H
 for f1 in bf[1]:
  q2=bf[2][(bf[2]>=f1)&(bf[2]<=f1+pd.Timedelta(hours=24))]
  if not len(q2):continue
  f2=q2[0];q4=bf[4][(bf[4]>=f2)&(bf[4]<=f2+pd.Timedelta(hours=24))]
  if len(q4):candidates['CASCADE124']=q4[0];break
 for r,ts in candidates.items():
  if ts is None or ts>dend:continue
  z=tf[4] if r.startswith('4H') else tf[4]
  # execution at first available hourly close at/after signal timestamp
  ix=p.index.searchsorted(ts,side='left')
  if ix>=len(p):continue
  px=float(p.close.iloc[ix]);tsex=p.index[ix];remain=max(0.,top/px-1.)
  # giveback from ex-post top to exit price; diagnostic only
  give=max(0.,1-px/top)
  # performance from entry to exit
  ret=px/float(x['entry'])-1
  out.append({'ci':ci,'rule':r,'ts':tsex,'remain':remain,'give':give,'ret':ret,'lag_daily':(dend-tsex).total_seconds()/3600})
R=pd.DataFrame(out);print('ROWS18B|%d'%len(R),flush=True)
for r in rules:
 q=R[R.rule==r]
 if q.empty:continue
 print('RULE18B|%s|n=%d|cov=%.3f|ret_med=%.4f|give_med=%.4f|remain_med=%.4f|gt5=%.3f|gt7.5=%.3f|gt10=%.3f|gt15=%.3f|daily_lead_med=%.1f'%(
  r,len(q),len(q)/len(O),np.median(q.ret),np.median(q.give),np.median(q.remain),np.mean(q.remain>.05),np.mean(q.remain>.075),np.mean(q.remain>.10),np.mean(q.remain>.15),np.nanmedian(q.lag_daily)),flush=True)
# pairwise on same cycles so coverage differences cannot fake superiority
for a,b in [('4H','4H_PERSIST8'),('4H','4H_6H'),('4H','CASCADE124')]:
 A=R[R.rule==a].set_index('ci');B=R[R.rule==b].set_index('ci');idx=A.index.intersection(B.index)
 if not len(idx):continue
 da=A.loc[idx];db=B.loc[idx]
 print('PAIR18B|%s_vs_%s|n=%d|ret_delta_med=%.4f|give_delta_med=%.4f|better_ret=%.3f'%(b,a,len(idx),np.median(db.ret-da.ret),np.median(db.give-da.give),np.mean(db.ret>da.ret)),flush=True)
print('DONE18B',flush=True)