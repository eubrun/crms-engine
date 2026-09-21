"""Phase 18A: descriptive multi-timeframe PSAR sequence around cycle TOP.

Uses the same causal intraday Daily-SAR BUY event construction as Phase 13B.
Extends each hourly path to 45d, finds the first subsequent Daily PSAR bearish flip,
then measures 1/2/4/6/8/12h bearish PSAR flips around the ex-post cycle TOP.
TOP is measurement-only and never an input to a live rule.
"""
exec(open('sar_phase8_presar_intraday.py').read().split("for th in [.45,.50,.55,.60]:")[0])

TF=[1,2,4,6,8,12]

def psar_state(df,step=.02,maxaf=.2):
 hi=df.high.to_numpy(float);lo=df.low.to_numpy(float);n=len(df)
 bull=np.ones(n,dtype=bool);sar=np.full(n,np.nan)
 if n==0:return bull,sar
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

# Pull longer 1h histories once per symbol. Phase8 only retained ~16d paths.
HIST={}
for sym in t.symbol.unique():
 r=t[t.symbol==sym]
 start=pd.Timestamp(r.index.min(),tz='UTC')-pd.Timedelta(days=2)
 end=pd.Timestamp(r.index.max(),tz='UTC')+pd.Timedelta(days=50)
 try:
  HIST[sym]=hourly(sym,int(start.timestamp()*1000),int(end.timestamp()*1000))
  print('HIST18A|%s|bars=%d'%(sym,len(HIST[sym])),flush=True)
 except Exception as e:
  HIST[sym]=pd.DataFrame();print('FAIL18A_HIST|%s|%s'%(sym,str(e)[:120]),flush=True)

# Daily PSAR SELL timestamps are reconstructed from the daily series itself.
DAILY={}
for sym in t.symbol.unique():
 try:
  d=fetch(sym);b,_=psar_state(d);sell=np.where((~b)&np.r_[True,b[:-1]])[0]
  DAILY[sym]=pd.DatetimeIndex(d.index[sell])
 except Exception as e:
  DAILY[sym]=pd.DatetimeIndex([]);print('FAIL18A_DAILY|%s|%s'%(sym,str(e)[:120]),flush=True)

cycles=[]
for ix,r in t.iterrows():
 sym=r.symbol;buy=pd.Timestamp(r['_path'].index[0]) if len(r['_path']) else pd.Timestamp(ix,tz='UTC')
 h=HIST.get(sym,pd.DataFrame())
 if h.empty:continue
 path=h[(h.index>=buy)&(h.index<buy+pd.Timedelta(days=45))].copy()
 if len(path)<48:continue
 sells=DAILY.get(sym,pd.DatetimeIndex([]));sells=sells[sells>buy]
 sell=sells[0] if len(sells) else pd.NaT
 end=sell+pd.Timedelta(days=1) if pd.notna(sell) else buy+pd.Timedelta(days=45)
 cyc=path[path.index<end]
 if cyc.empty:continue
 k=int(np.argmax(cyc.high.to_numpy(float)));top=cyc.index[k];toppx=float(cyc.high.iloc[k]);entry=float(r.entry);mfe=toppx/entry-1
 rec={'symbol':sym,'buy':buy,'sell':sell,'top':top,'entry':entry,'toppx':toppx,'mfe':mfe,'path':path}
 cycles.append(rec)
print('CYCLES18A|n=%d|sellcov=%.3f|buy_sell_med_days=%.2f|buy_top_med_days=%.2f|top_sell_med_days=%.2f'%(
 len(cycles),np.mean([pd.notna(c['sell']) for c in cycles]),
 np.nanmedian([(c['sell']-c['buy']).total_seconds()/86400 if pd.notna(c['sell']) else np.nan for c in cycles]),
 np.nanmedian([(c['top']-c['buy']).total_seconds()/86400 for c in cycles]),
 np.nanmedian([(c['sell']-c['top']).total_seconds()/86400 if pd.notna(c['sell']) else np.nan for c in cycles])),flush=True)
for x in [.10,.20,.30,.50]:print('MFE18A|%.2f|%.3f'%(x,np.mean([c['mfe']>=x for c in cycles])),flush=True)

rows=[]
for ci,c in enumerate(cycles):
 p=c['path'];w0=c['top']-pd.Timedelta(hours=72);w1=c['top']+pd.Timedelta(hours=72)
 for H in TF:
  z=p if H==1 else p.resample('%dh'%H,origin='epoch',label='right',closed='right').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
  b,_=psar_state(z);fi=np.where((~b)&np.r_[True,b[:-1]])[0];flips=z.index[fi];flips=flips[(flips>=w0)&(flips<=w1)]
  if not len(flips):continue
  f=flips[0];after=p[p.index>=f];after_sell=after if pd.isna(c['sell']) else after[after.index<=c['sell']]
  newhi=bool(len(after_sell) and after_sell.high.max()>c['toppx']*1.001)
  add=max(0,float(after_sell.high.max()/c['toppx']-1)) if len(after_sell) else np.nan
  dd=float(after_sell.low.min()/c['toppx']-1) if len(after_sell) else np.nan
  # first bullish re-flip after selected bearish flip and before Daily SELL
  bi=np.where(b&np.r_[False,~b[:-1]])[0];bullflips=z.index[bi];bf=bullflips[bullflips>f]
  if pd.notna(c['sell']):bf=bf[bf<c['sell']]
  rows.append({'ci':ci,'symbol':c['symbol'],'H':H,'mfe':c['mfe'],'flip':f,'to_top_h':(c['top']-f).total_seconds()/3600,'to_sell_h':(c['sell']-f).total_seconds()/3600 if pd.notna(c['sell']) else np.nan,'new_high':newhi,'add_up':add,'dd':dd,'reflip':len(bf)>0})
R=pd.DataFrame(rows)
print('ROWS18A|n=%d'%len(R),flush=True)
for H in TF:
 q=R[R.H==H];den=len(cycles)
 if q.empty:continue
 print('TF18A|%dh|coverage=%.3f|pretop=%.3f|leadtop_med=%.1f|leadsell_med=%.1f|falsehi=%.3f|reflip=%.3f|addup_med=%.4f|dd_med=%.4f'%(
  H,len(q)/den,np.mean(q.to_top_h>0),np.median(q.to_top_h),np.nanmedian(q.to_sell_h),q.new_high.mean(),q.reflip.mean(),np.nanmedian(q.add_up),np.nanmedian(q.dd)),flush=True)

bins=[-np.inf,.10,.20,.30,.50,np.inf];labels=['<10','10-20','20-30','30-50','50+']
R['bucket']=pd.cut(R.mfe,bins=bins,labels=labels,right=False)
for (bucket,H),q in R.groupby(['bucket','H'],observed=True):
 print('BUCKET18A|%s|%dh|n=%d|pretop=%.3f|leadtop=%.1f|leadsell=%.1f|falsehi=%.3f|reflip=%.3f'%(
  bucket,H,len(q),np.mean(q.to_top_h>0),np.median(q.to_top_h),np.nanmedian(q.to_sell_h),q.new_high.mean(),q.reflip.mean()),flush=True)

# Ordered sequence signature using first selected flip per TF in the TOP window.
seq={}
for ci,q in R.groupby('ci'):
 qq=q.sort_values(['flip','H']);sig='>'.join(str(int(x)) for x in qq.H.to_list())
 seq[sig]=seq.get(sig,0)+1
for sig,n in sorted(seq.items(),key=lambda x:x[1],reverse=True)[:25]:print('SEQ18A|%s|n=%d|freq=%.3f'%(sig,n,n/max(1,len(cycles))),flush=True)
print('DONE18A',flush=True)
