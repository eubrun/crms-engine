"""Phase18C: causal combinatorial SAR exit optimizer on exact Phase17C OOS.
Discovery/validation split is chronological inside the 1081 OOS events (70/30).
TOP is evaluation only. Candidate rules are generated mechanically from SAR flip counts,
TF confirmations, simultaneous bearish sets and persistence. Selection score prioritizes
captured return while penalizing >5/7.5/10/15% missed upside, low coverage and complexity.
"""
exec(open('sar_phase18b_exit_rules.py').read().split("H={}")[0])
EV=sorted(EV,key=lambda x:x['ts']);O=EV[int(len(EV)*.20):]
print('OOS18C|events=%d|oos=%d'%(len(EV),len(O)),flush=True)
TFS=[1,2,4,6,8,12]
H={}
for sym in sorted(set(x['sym'] for x in O)):
 xs=[x for x in O if x['sym']==sym];a=min(x['ts'] for x in xs)-pd.Timedelta(days=4);e=max(x['ts'] for x in xs)+pd.Timedelta(hours=1081)
 try:H[sym]=hourly(sym,int(a.timestamp()*1000),int(e.timestamp()*1000));print('H18C|%s|%d'%(sym,len(H[sym])),flush=True)
 except Exception as ex:H[sym]=pd.DataFrame();print('FAIL18C|%s|%s'%(sym,str(ex)[:120]),flush=True)

def first_after(arr,t,limit=None):
 q=arr[arr>=t]
 if limit is not None:q=q[q<=t+pd.Timedelta(hours=limit)]
 return q[0] if len(q) else None

def state_at(z,b,t):
 j=z.index.searchsorted(t,side='right')-1
 return bool(b[j]) if 0<=j<len(b) else True

# rule schema: name, complexity, trigger function returning timestamp or None
RULES=[]
# nth bearish flip of each TF
for tf0 in [2,4,6,8,12]:
 for nth in [1,2,3]:
  RULES.append(('F%d_%d'%(tf0,nth),1+nth*.25,('nth',tf0,nth)))
# nth 4H then confirmation on slower TF, lags
for nth in [1,2,3]:
 for cf in [6,8,12]:
  for lag in [12,24,48,72]:RULES.append(('F4_%d_TO_%d_L%d'%(nth,cf,lag),2+nth*.25,('seq',4,nth,cf,lag)))
# 4H nth flip plus simultaneous bearish state sets at/after trigger within wait
sets=[(6,),(8,),(12,),(6,8),(6,12),(8,12),(6,8,12)]
for nth in [1,2,3]:
 for ss in sets:
  for wait in [0,12,24,48]:RULES.append(('F4_%d_SET%s_W%d'%(nth,''.join(map(str,ss)),wait),2+len(ss)+nth*.25,('set',nth,ss,wait)))
# persistence of nth 4H bearish state
for nth in [1,2,3]:
 for ph in [4,8,12,24,36]:RULES.append(('F4_%d_P%d'%(nth,ph),1.5+nth*.25,('persist',nth,ph)))
print('RULESPACE18C|n=%d'%len(RULES),flush=True)
rows=[]
for ci,x in enumerate(O):
 p=H.get(x['sym'],pd.DataFrame());p=p[(p.index>=x['ts'])&(p.index<x['ts']+pd.Timedelta(hours=1081))]
 if p.empty:continue
 dend=x['sell'] if x['sell'] is not None else p.index[-1];cyc=p[p.index<=dend]
 if cyc.empty:continue
 top=float(cyc.high.max());zz={tf0:bars(p,tf0) for tf0 in TFS};bb={};ff={}
 for tf0 in TFS:bb[tf0],ff[tf0]=flips(zz[tf0])
 for name,cpl,spec in RULES:
  ts=None;typ=spec[0]
  if typ=='nth':
   _,tf0,n=spec
   if len(ff[tf0])>=n:ts=ff[tf0][n-1]
  elif typ=='seq':
   _,tf0,n,cf,lag=spec
   if len(ff[tf0])>=n:ts=first_after(ff[cf],ff[tf0][n-1],lag)
  elif typ=='persist':
   _,n,ph=spec
   if len(ff[4])>=n:
    f=ff[4][n-1];chk=f+pd.Timedelta(hours=ph)
    if chk<=dend and not state_at(zz[4],bb[4],chk):ts=chk
  elif typ=='set':
   _,n,ss,wait=spec
   if len(ff[4])>=n:
    f=ff[4][n-1]
    # search causal hourly grid until all requested TFs bearish
    for chk in p.index[(p.index>=f)&(p.index<=min(dend,f+pd.Timedelta(hours=wait)))]:
     if all(not state_at(zz[k],bb[k],chk) for k in ss):ts=chk;break
  if ts is None or ts>dend:continue
  ix=p.index.searchsorted(ts,side='left')
  if ix>=len(p):continue
  px=float(p.close.iloc[ix]);remain=max(0.,top/px-1.);give=max(0.,1-px/top);ret=px/float(x['entry'])-1
  rows.append((ci,name,cpl,ret,remain,give,(dend-p.index[ix]).total_seconds()/3600))
R=pd.DataFrame(rows,columns=['ci','rule','complexity','ret','remain','give','daily_lead'])
print('ROWS18C|n=%d'%len(R),flush=True)
# chronological inner split by event index: 70% discovery, 30% untouched validation
cut=int(len(O)*.70);TR=R[R.ci<cut];VA=R[R.ci>=cut]
def metrics(q,total):
 return {'n':len(q),'cov':len(q)/total,'ret':np.median(q.ret) if len(q) else np.nan,'give':np.median(q.give) if len(q) else np.nan,'rem':np.median(q.remain) if len(q) else np.nan,'g5':np.mean(q.remain>.05) if len(q) else np.nan,'g75':np.mean(q.remain>.075) if len(q) else np.nan,'g10':np.mean(q.remain>.10) if len(q) else np.nan,'g15':np.mean(q.remain>.15) if len(q) else np.nan}
def score(m,c):
 if m['n']<80 or m['cov']<.35:return -999
 # utility: reward realized return; heavily penalize meaningful missed upside, low coverage, complexity
 return m['ret']-0.40*m['g5']-0.25*m['g75']-0.20*m['g10']-0.15*m['g15']-0.10*(1-m['cov'])-0.002*c
rank=[]
for name,cpl,_ in RULES:
 m=metrics(TR[TR.rule==name],cut);rank.append((score(m,cpl),name,cpl,m))
rank.sort(reverse=True,key=lambda x:x[0])
for j,(sc,name,cpl,m) in enumerate(rank[:20],1):print('TRAIN18C|rank=%d|%s|score=%.4f|n=%d|cov=%.3f|ret=%.4f|give=%.4f|rem=%.4f|gt5=%.3f|gt75=%.3f|gt10=%.3f|gt15=%.3f|cpl=%.2f'%(j,name,sc,m['n'],m['cov'],m['ret'],m['give'],m['rem'],m['g5'],m['g75'],m['g10'],m['g15'],cpl),flush=True)
# untouched validation only for top 10 selected on training
for j,(sc,name,cpl,mt) in enumerate(rank[:10],1):
 mv=metrics(VA[VA.rule==name],len(O)-cut)
 print('VALID18C|trainrank=%d|%s|n=%d|cov=%.3f|ret=%.4f|give=%.4f|rem=%.4f|gt5=%.3f|gt75=%.3f|gt10=%.3f|gt15=%.3f|daily_lead=%.1f'%(j,name,mv['n'],mv['cov'],mv['ret'],mv['give'],mv['rem'],mv['g5'],mv['g75'],mv['g10'],mv['g15'],np.nanmedian(VA[VA.rule==name].daily_lead) if mv['n'] else np.nan),flush=True)
print('DONE18C',flush=True)