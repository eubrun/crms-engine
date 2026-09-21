"""Phase18A.1: exact Phase17C OOS universe + causal multi-TF SAR warning cost."""
exec(open('sar_phase17c_cycle_integrity.py').read().split("EV=sorted(EV,key=lambda x:x['ts']);cut=int(len(EV)*.20);O=EV[cut:]")[0])
EV=sorted(EV,key=lambda x:x['ts']);cut=int(len(EV)*.20);O=EV[cut:]
print('OOS18A1|events=%d|oos=%d'%(len(EV),len(O)),flush=True)
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
H={}
for sym in sorted(set(x['sym'] for x in O)):
 xs=[x for x in O if x['sym']==sym];a=min(x['ts'] for x in xs)-pd.Timedelta(days=4);b=max(x['ts'] for x in xs)+pd.Timedelta(hours=1081)
 try:H[sym]=hourly(sym,int(a.timestamp()*1000),int(b.timestamp()*1000));print('H18A1|%s|%d'%(sym,len(H[sym])),flush=True)
 except Exception as e:H[sym]=pd.DataFrame();print('FAIL18A1|%s|%s'%(sym,str(e)[:120]),flush=True)
rows=[];seq={}
for ci,x in enumerate(O):
 h=H.get(x['sym'],pd.DataFrame());p=h[(h.index>=x['ts'])&(h.index<x['ts']+pd.Timedelta(hours=1081))]
 if p.empty:continue
 end=x['sell'] if x['sell'] is not None else p.index[-1];cyc=p[p.index<=end]
 if cyc.empty:continue
 top=cyc.index[int(np.argmax(cyc.high.to_numpy(float)))];toppx=float(cyc.high.max());w0=top-pd.Timedelta(hours=72);w1=top+pd.Timedelta(hours=72);sig=[]
 for tf in TF:
  z=p if tf==1 else p.resample('%dh'%tf,origin='epoch',label='right',closed='right').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
  b,_=psar_state(z);ii=np.where((~b)&np.r_[True,b[:-1]])[0];fl=z.index[ii];fl=fl[(fl>=w0)&(fl<=w1)]
  if not len(fl):continue
  f=fl[0];sig.append((f,tf));sp=float(z.loc[f].close);remain=max(0.,toppx/sp-1.);lead=(top-f).total_seconds()/3600;ds=(x['sell']-f).total_seconds()/3600 if x['sell'] is not None else np.nan
  bi=np.where(b&np.r_[False,~b[:-1]])[0];bf=z.index[bi];bf=bf[bf>f];bf=bf if x['sell'] is None else bf[bf<x['sell']]
  aft=cyc[cyc.index>f];higher=bool(len(aft) and aft.high.max()>=sp*1.02)
  rows.append({'ci':ci,'tf':tf,'mfe':x['mfe'],'lead':lead,'ds':ds,'remain':remain,'higher2':higher,'reflip':len(bf)>0})
 if sig:
  s='>'.join(str(tf) for _,tf in sorted(sig));seq[s]=seq.get(s,0)+1
R=pd.DataFrame(rows);print('ROWS18A1|%d'%len(R),flush=True)
for tf in TF:
 q=R[R['tf']==tf]
 if len(q):print('TF18A1|%dh|n=%d|cov=%.3f|pretop=%.3f|lead_med=%.1f|daily_lead_med=%.1f|remain_med=%.4f|remain_p75=%.4f|higher2=%.3f|reflip=%.3f'%(tf,len(q),len(q)/len(O),np.mean(q.lead>0),np.median(q.lead),np.nanmedian(q.ds),np.median(q.remain),np.quantile(q.remain,.75),q.higher2.mean(),q.reflip.mean()),flush=True)
for s,n in sorted(seq.items(),key=lambda a:a[1],reverse=True)[:15]:print('SEQ18A1|%s|n=%d|freq=%.3f'%(s,n,n/len(O)),flush=True)
print('DONE18A1',flush=True)