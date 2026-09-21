"""Phase18D: causal running-MFE x SAR optimizer. Exact 1081 Phase17C OOS; chronological 702/379 split."""
exec(open('sar_phase18c_optimizer.py').read().split("# Candidate rule descriptors")[0])
# Override: same OOS/split/TF/helpers already built above
TH=[.05,.075,.10,.15,.20,.30,.50]
# rules: threshold, SAR tf, nth bearish flip occurring only AFTER threshold was first reached
rules=[]
for th in TH:
 for tf in TF:
  for nth in [1,2,3]:rules.append(('MFE%d__%dH_F%d'%(round(th*1000),tf,nth),th,tf,nth,1+nth*.15))
print('RULES18D|%d'%len(rules),flush=True)
H={}
for sym in sorted(set(x['sym'] for x in O)):
 xs=[x for x in O if x['sym']==sym];a=min(x['ts'] for x in xs)-pd.Timedelta(days=4);e=max(x['ts'] for x in xs)+pd.Timedelta(hours=1081)
 try:H[sym]=hourly(sym,int(a.timestamp()*1000),int(e.timestamp()*1000));print('H18D|%s|%d'%(sym,len(H[sym])),flush=True)
 except Exception as ex:H[sym]=pd.DataFrame();print('FAIL18D|%s|%s'%(sym,str(ex)[:100]),flush=True)
out=[]
for ci,x in enumerate(O):
 p=H.get(x['sym'],pd.DataFrame());p=p[(p.index>=x['ts'])&(p.index<x['ts']+pd.Timedelta(hours=1081))]
 if p.empty:continue
 dend=x['sell'] if x['sell'] is not None else p.index[-1];cyc=p[p.index<=dend]
 if cyc.empty:continue
 entry=float(x['entry']);top=float(cyc.high.max());Z={};F={}
 for tf in TF:
  z=bars(p,tf);b,_=psar_state(z);F[tf]=z.index[np.where((~b)&np.r_[True,b[:-1]])[0]];Z[tf]=z
 # causal first time each running MFE threshold is actually observed
 hit={}
 runhi=cyc.high.cummax()/entry-1
 for th in TH:
  q=runhi[runhi>=th];hit[th]=q.index[0] if len(q) else None
 for name,th,tf,nth,cx in rules:
  h=hit[th]
  if h is None:continue
  ff=F[tf][F[tf]>=h]
  if len(ff)<nth:continue
  ts=ff[nth-1]
  if ts>dend:continue
  ix=p.index.searchsorted(ts,'left')
  if ix>=len(p):continue
  px=float(p.close.iloc[ix]);remain=max(0.,top/px-1.);give=max(0.,1-px/top);ret=px/entry-1
  out.append((ci,name,remain,give,ret,(dend-p.index[ix]).total_seconds()/3600,cx,th,tf,nth))
R=pd.DataFrame(out,columns=['ci','rule','remain','give','ret','daily_lead','complexity','th','tf','nth']);print('ROWS18D|%d'%len(R),flush=True)
def metrics(q,den):
 return {'n':len(q),'cov':len(q)/den,'ret':np.median(q.ret) if len(q) else np.nan,'give':np.median(q.give) if len(q) else np.nan,'rem':np.median(q.remain) if len(q) else np.nan,'g5':np.mean(q.remain>.05) if len(q) else 1.,'g75':np.mean(q.remain>.075) if len(q) else 1.,'g10':np.mean(q.remain>.10) if len(q) else 1.,'g15':np.mean(q.remain>.15) if len(q) else 1.}
def score(m,cx):return 3*m['g5']+2*m['g75']+1.5*m['g10']+m['g15']+1.25*(1-m['cov'])+.35*m['give']-.25*m['ret']+.015*cx
rank=[]
for name,th,tf,nth,cx in rules:
 q=R[(R.rule==name)&(R.ci<split)];m=metrics(q,split);rank.append((score(m,cx),name,m))
rank.sort()
for k,(sc,name,m) in enumerate(rank[:25],1):print('TRAIN18D|%d|%s|score=%.4f|n=%d|cov=%.3f|ret=%.4f|give=%.4f|rem=%.4f|gt5=%.3f|gt7.5=%.3f|gt10=%.3f|gt15=%.3f'%(k,name,sc,m['n'],m['cov'],m['ret'],m['give'],m['rem'],m['g5'],m['g75'],m['g10'],m['g15']),flush=True)
for k,(sc,name,mt) in enumerate(rank[:25],1):
 q=R[(R.rule==name)&(R.ci>=split)];m=metrics(q,N-split)
 print('TEST18D|%d|%s|n=%d|cov=%.3f|ret=%.4f|give=%.4f|rem=%.4f|gt5=%.3f|gt7.5=%.3f|gt10=%.3f|gt15=%.3f|daily_lead=%.1f'%(k,name,m['n'],m['cov'],m['ret'],m['give'],m['rem'],m['g5'],m['g75'],m['g10'],m['g15'],np.nanmedian(q.daily_lead) if len(q) else np.nan),flush=True)
print('DONE18D',flush=True)