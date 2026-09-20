"""Phase16B standalone: conditional MFE exit tournament."""
# Load Phase13B data preparation only.
exec(open('sar_phase13b_exit_fast.py').read().split("# 90+ meaningful configurations")[0])

def ema(x,n):
 a=2.0/(n+1);out=np.empty(len(x));out[0]=x[0]
 for i in range(1,len(x)):out[i]=a*x[i]+(1-a)*out[i-1]
 return out

def atr(h,l,c,n):
 tr=np.empty(len(c));tr[0]=h[0]-l[0]
 for i in range(1,len(c)):tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
 out=np.empty(len(c));out[0]=tr[0];a=1.0/n
 for i in range(1,len(c)):out[i]=(1-a)*out[i-1]+a*tr[i]
 return out

def exit_idx(d,rule):
 typ,p=rule;c=d['cl'];h=d['hi'];l=d['lo'];ts=d['ts'];n=min(len(c),240)
 if typ=='SAR':
  fs=d['f%d'%p]
  for j in range(1,n):
   if any((f<=ts[j]) and (f>ts[j-1]) for f in fs):return j
 elif typ=='TRAIL':
  pk=h[0]
  for j in range(1,n):
   pk=max(pk,h[j])
   if c[j]<=pk*(1-p):return j
 elif typ=='CHAND':
  period,k=p;A=atr(h[:n],l[:n],c[:n],period);pk=h[0]
  for j in range(max(2,period),n):
   pk=max(pk,h[j])
   if c[j]<=pk-k*A[j]:return j
 elif typ=='LOW':
  for j in range(p,n):
   if c[j]<np.min(l[j-p:j]):return j
 elif typ=='EMA':
  E=ema(c[:n],p)
  for j in range(max(2,p),n):
   if c[j]<E[j]:return j
 return n-1

RULES=[]
for H in TF:RULES.append(('SAR',H))
for q in [.02,.03,.04,.05,.06,.08,.10,.12,.15]:RULES.append(('TRAIL',q))
for per in [12,24,48]:
 for k in [1.5,2.,2.5,3.,3.5,4.]:RULES.append(('CHAND',(per,k)))
for w in [6,12,24,48,72,120]:RULES.append(('LOW',w))
for w in [6,12,24,48,72,120]:RULES.append(('EMA',w))
et=np.array([pd.Timestamp(d['ix']).value for d in EV]);order=np.argsort(et);cuts=np.linspace(0,len(order),6,dtype=int);base=order[cuts[1]:]
print('TOURN16B|rules=%d|events=%d'%(len(RULES),len(base)),flush=True)
for threshold in [.05,.10,.20,.30]:
 ids=np.array([ei for ei in base if EV[ei]['mfe']>=threshold],dtype=int);print('BUCKET16B|mfe>=%.2f|n=%d'%(threshold,len(ids)),flush=True);R=[]
 for r in RULES:
  vals=[]
  for ei in ids:
   d=EV[ei];e=d['e'];j=exit_idx(d,r);px=d['cl'][j];real=px/e-1-.003;mfe=d['mfe'];cap=real/mfe;gbr=max(0,(mfe-real)/mfe);pk=max(e,np.max(d['hi'][:j+1]));future=np.max(d['hi'][j+1:min(len(d['hi']),240)]) if j+1<min(len(d['hi']),240) else px;early=float(future>=pk*1.02);vals.append((real,cap,gbr,j/24.,early))
  a=np.asarray(vals);s={'mean':a[:,0].mean(),'med':np.median(a[:,0]),'win':(a[:,0]>0).mean(),'cap':np.median(a[:,1]),'avgcap':a[:,1].mean(),'gbr':np.median(a[:,2]),'days':np.median(a[:,3]),'early':a[:,4].mean(),'worst':a[:,0].min()};score=s['cap']+.25*s['mean']-.05*s['early'];R.append((score,r,s))
 for rank,(score,r,s) in enumerate(sorted(R,key=lambda x:x[0],reverse=True)[:15],1):print('TOP16B|mfe%.2f|%d|%s|%s|n%d|score%.3f|mean%.4f|med%.4f|win%.3f|medcap%.3f|avgcap%.3f|medGBR%.3f|early%.3f|days%.2f|worst%.4f'%(threshold,rank,r[0],str(r[1]),len(ids),score,s['mean'],s['med'],s['win'],s['cap'],s['avgcap'],s['gbr'],s['early'],s['days'],s['worst']),flush=True)
 for fam in ['SAR','TRAIL','CHAND','LOW','EMA']:
  q=max((x for x in R if x[1][0]==fam),key=lambda x:x[0]);s=q[2];print('FAM16B|mfe%.2f|%s|%s|mean%.4f|win%.3f|cap%.3f|GBR%.3f|early%.3f|days%.2f'%(threshold,fam,str(q[1][1]),s['mean'],s['win'],s['cap'],s['gbr'],s['early'],s['days']),flush=True)
print('DONE16B',flush=True)
