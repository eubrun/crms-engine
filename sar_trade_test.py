import numpy as np
from crms import ASSETS,fetch,indicators

def test(symbol):
 d=fetch(symbol); z=indicators(d).dropna(subset=['psar']).copy()
 bull=z['psar_bull'].astype(bool)
 entry=bull & ~bull.shift(1,fill_value=False)
 exit_=~bull & bull.shift(1,fill_value=False)
 ep=list(np.flatnonzero(entry.to_numpy())); xp=list(np.flatnonzero(exit_.to_numpy()))
 trades=[]
 for i in ep:
  js=[j for j in xp if j>i]
  if not js: continue
  j=js[0]; a=float(z.close.iloc[i]); b=float(z.close.iloc[j]); r=b/a-1
  trades.append((i,j,r,j-i))
 return z,trades
alltr=[]
print('METHOD|entry=close on daily flip price>SAR|exit=close on next daily flip price<SAR|fees=0|slippage=0',flush=True)
for s in ASSETS:
 try:
  z,t=test(s); alltr += [(s,*q) for q in t]
  r=np.array([q[2] for q in t],float)
  if len(r): print('ASSET|%s|start=%s|end=%s|trades=%d|win=%.4f|mean=%.4f|median=%.4f|compound=%.4f|avg_days=%.2f'%(s,z.index.min().date(),z.index.max().date(),len(r),(r>0).mean(),r.mean(),np.median(r),np.prod(1+r)-1,np.mean([q[3] for q in t])),flush=True)
 except Exception as e: print('FAIL|%s|%s'%(s,str(e)[:120]),flush=True)
r=np.array([q[3] for q in alltr],float)
print('TOTAL|assets=%d|trades=%d|win=%.4f|mean=%.4f|median=%.4f|compound_all_independent=%.4f|avg_days=%.2f'%(len(set(q[0] for q in alltr)),len(r),(r>0).mean(),r.mean(),np.median(r),np.prod(1+r)-1,np.mean([q[4] for q in alltr])),flush=True)
