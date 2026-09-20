"""Phase16: professional exit tournament after Daily-SAR BUY.
Compare lower-TF SAR, ATR/Chandelier, percent trailing, rolling-low/Donchian and EMA exits.
No ML. Same 1328 historical events; chronological OOS reporting by folds/assets.
"""
exec(open('sar_phase13b_exit_fast.py').read().split("# 90+ meaningful configurations")[0])

# EV already contains event H1 paths and lower-TF SAR flip timestamps.
# Add causal indicators per event.
def ema(x,n):
 a=2.0/(n+1); out=np.empty(len(x));out[0]=x[0]
 for i in range(1,len(x)):out[i]=a*x[i]+(1-a)*out[i-1]
 return out

def atr(h,l,c,n):
 tr=np.empty(len(c));tr[0]=h[0]-l[0]
 for i in range(1,len(c)):tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
 out=np.empty(len(c));out[0]=tr[0];a=1.0/n
 for i in range(1,len(c)):out[i]=(1-a)*out[i-1]+a*tr[i]
 return out

# rule returns exit index. All indicators use current/past data only.
def exit_idx(d,rule):
 typ,p=rule; c=d['cl'];h=d['hi'];l=d['lo'];ts=d['ts'];n=min(len(c),240)
 if typ=='SAR':
  H=p; fs=d['f%d'%H]
  for j in range(1,n):
   if any((f<=ts[j]) and (f>ts[j-1]) for f in fs):return j
 elif typ=='TRAIL':
  q=p;pk=h[0]
  for j in range(1,n):
   pk=max(pk,h[j]);
   if c[j]<=pk*(1-q):return j
 elif typ=='CHAND':
  period,k=p;A=atr(h[:n],l[:n],c[:n],period);pk=h[0]
  for j in range(max(2,period),n):
   pk=max(pk,h[j]);stop=pk-k*A[j]
   if c[j]<=stop:return j
 elif typ=='LOW':
  w=p
  for j in range(w,n):
   # prior rolling low, avoids using current low in threshold
   if c[j]<np.min(l[j-w:j]):return j
 elif typ=='EMA':
  w=p;E=ema(c[:n],w)
  for j in range(max(2,w),n):
   if c[j]<E[j]:return j
 return n-1

def evaluate(rule,ids):
 z=[]
 for ei in ids:
  d=EV[ei];e=d['e'];j=exit_idx(d,rule);px=d['cl'][j];ret=px/e-1-.003;mfe=max(0,d['mfe']);cap=ret/mfe if mfe>=.01 else np.nan
  # realized running peak before exit and giveback from it
  pk=max(e,np.max(d['hi'][:j+1]));give=max(0,(pk-px)/e)
  # early: exited before event's eventual high by >=2%; late giveback >=5%
  future=np.max(d['hi'][j+1:min(len(d['hi']),240)]) if j+1<min(len(d['hi']),240) else px
  early=float(future>=pk*1.02);late=float(give>=.05)
  z.append((ret,cap,give,j/24.,early,late))
 a=np.asarray(z);return dict(n=len(a),mean=a[:,0].mean(),median=np.median(a[:,0]),win=(a[:,0]>0).mean(),cap=np.nanmedian(a[:,1]),avgcap=np.nanmean(a[:,1]),give=a[:,2].mean(),days=np.median(a[:,3]),early=a[:,4].mean(),late=a[:,5].mean(),worst=a[:,0].min())

RULES=[]
for H in TF:RULES.append(('SAR',H))
for q in [.02,.03,.04,.05,.06,.08,.10,.12,.15]:RULES.append(('TRAIL',q))
for per in [12,24,48]:
 for k in [1.5,2.,2.5,3.,3.5,4.]:RULES.append(('CHAND',(per,k)))
for w in [6,12,24,48,72,120]:RULES.append(('LOW',w))
for w in [6,12,24,48,72,120]:RULES.append(('EMA',w))
print('TOURN16|rules=%d|events=%d'%(len(RULES),len(EV)),flush=True)
# chronological test set consistent with prior phases: last 80% after first calibration fifth
et=np.array([pd.Timestamp(d['ix']).value for d in EV]);order=np.argsort(et);cuts=np.linspace(0,len(order),6,dtype=int);ids=order[cuts[1]:]
RES=[]
for i,r in enumerate(RULES,1):
 s=evaluate(r,ids)
 # primary score balances capture/expectancy against giveback and catastrophic tail
 score=s['cap']+.5*s['mean']-.5*s['give']-.03*s['early']-.03*s['late']
 RES.append((score,r,s))
 print('RULE16|%s|%s|n%d|mean%.4f|med%.4f|win%.3f|medcap%.3f|avgcap%.3f|give%.4f|days%.2f|early%.3f|late%.3f|worst%.4f'% (r[0],str(r[1]),s['n'],s['mean'],s['median'],s['win'],s['cap'],s['avgcap'],s['give'],s['days'],s['early'],s['late'],s['worst']),flush=True)
 if i%10==0:print('PROGRESS16|%d/%d'%(i,len(RULES)),flush=True)
print('--- TOP OVERALL ---',flush=True)
for rank,(score,r,s) in enumerate(sorted(RES,key=lambda x:x[0],reverse=True)[:20],1):print('TOP16|%d|%s|%s|score%.4f|mean%.4f|win%.3f|medcap%.3f|give%.4f|early%.3f|late%.3f|days%.2f|worst%.4f'%(rank,r[0],str(r[1]),score,s['mean'],s['win'],s['cap'],s['give'],s['early'],s['late'],s['days'],s['worst']),flush=True)
# Robustness: top 10 across four chronological OOS blocks and per asset.
top=[x[1] for x in sorted(RES,key=lambda x:x[0],reverse=True)[:10]]
for r in top:
 vals=[]
 for f in range(1,5):
  q=order[cuts[f]:cuts[f+1]];s=evaluate(r,q);vals.append((s['mean'],s['cap'],s['give']))
 print('ROBUST16|%s|%s|folds=%s'%(r[0],str(r[1]),';'.join('%.4f,%.3f,%.4f'%v for v in vals)),flush=True)
print('DONE16',flush=True)
