"""Phase 9: isolate pre-SAR geometry without shrinking sample by arbitrary HGB threshold.
Rebuild Phase8 events with 1h trigger; evaluate simple pre-known geometry rules and nested walk-forward rule selection.
Goal: >=100 OOS trades, improve win rate/expectancy, worst controlled. No end-of-day cross info.
"""
exec(open('sar_phase8_presar_intraday.py').read().split("for th in [.45,.50,.55,.60]:")[0])
# t now contains events + OOS prob. Fixed execution: TP15/SL6/H5 from robust Phase8 region.
def ex(r): return sim(r,.15,.06,5)[0]
t['ret']=t.apply(ex,axis=1)
# geometry families, all values known BEFORE trigger. Broad ranges only, no fine grid.
rules=[]
for dmax in [.02,.04,.06,.10,.15]:
 for amin,amax in [(-.02,.03),(0,.05),(.01,.08),(.02,.12)]:
  for conv in [0,1]:
   rules.append((dmax,amin,amax,conv))
def mask(df,r):
 dmax,lo,hi,conv=r
 m=(df.d1>=0)&(df.d1<=dmax)&(df.approach5>=lo)&(df.approach5<=hi)
 if conv:m&=df.conv3.eq(1)
 return m
def stats(g):
 x=g.ret.dropna();return (len(x),(x>0).mean() if len(x) else np.nan,x.mean() if len(x) else np.nan,x.median() if len(x) else np.nan,x.min() if len(x) else np.nan)
# descriptive full OOS table with minimum sample constraints
for pth in [.35,.40,.45,.50]:
 best=[]
 base=t[t.prob>=pth]
 for r in rules:
  g=base[mask(base,r)];n,wr,me,md,wo=stats(g)
  if n>=60:
   score=me+.03*wr+.15*wo;best.append((score,r,n,wr,me,md,wo))
 for rank,q in enumerate(sorted(best,reverse=True)[:10],1):
  sc,r,n,wr,me,md,wo=q;print('GEO9|pth=%.2f|rank=%d|dmax=%.3f|a5=%.3f:%.3f|conv=%d|n=%d|win=%.3f|mean=%.4f|med=%.4f|worst=%.4f|score=%.4f'%(pth,rank,r[0],r[1],r[2],r[3],n,wr,me,md,wo,sc),flush=True)
# nested selection. For each fold 2..5 choose pth+geometry using PRIOR OOS only; enforce >=25 historical trades and favor sample via weak penalty against tiny sets.
idx=pd.DatetimeIndex(pd.to_datetime(t.index,utc=True)).tz_convert(None);D=idx.unique().sort_values();bounds=[.50,.60,.70,.80,.90,1.0];allr=[];alln=0
for f in range(1,5):
 h0=D[int(len(D)*.50)];h1=D[int(len(D)*bounds[f])];q0=h1;q1=D[-1]+pd.Timedelta(days=1) if f==4 else D[int(len(D)*bounds[f+1])]
 hist=t[(idx>=h0)&(idx<h1)];test=t[(idx>=q0)&(idx<q1)];best=None
 for pth in [.35,.40,.45,.50]:
  hb=hist[hist.prob>=pth]
  for r in rules:
   g=hb[mask(hb,r)];n,wr,me,md,wo=stats(g)
   if n<25:continue
   score=me+.03*wr+.15*wo + .002*min(n,100)/100
   if best is None or score>best[0]:best=(score,pth,r,n)
 if best is None:continue
 _,pth,r,hn=best;g=test[(test.prob>=pth)&mask(test,r)];n,wr,me,md,wo=stats(g);allr+=g.ret.dropna().tolist();alln+=n
 print('NEST9|fold=%d|pth=%.2f|dmax=%.3f|a5=%.3f:%.3f|conv=%d|histn=%d|n=%d|win=%.3f|mean=%.4f|worst=%.4f'%(f+1,pth,r[0],r[1],r[2],r[3],hn,n,wr,me,wo),flush=True)
if allr:
 a=np.array(allr);print('NESTSUM9|n=%d|win=%.3f|mean=%.4f|median=%.4f|worst=%.4f'%(len(a),(a>0).mean(),a.mean(),np.median(a),a.min()),flush=True)
# direct interpretable geometry diagnostics independent of HGB
for dmax in [.02,.04,.06,.10]:
 g=t[(t.d1>=0)&(t.d1<=dmax)];n,wr,me,md,wo=stats(g);print('DIST9|dmax=%.3f|n=%d|win=%.3f|mean=%.4f|worst=%.4f'%(dmax,n,wr,me,wo),flush=True)
print('DONE9',flush=True)
