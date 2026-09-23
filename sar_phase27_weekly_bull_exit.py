"""Phase27: For fixed Daily+ entries whose causal Weekly PSAR is bullish, determine whether first 8H bearish should SELL or HOLD to 12H/Daily. Nested walk-forward, direct PnL."""
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
exec(open('sar_phase18c_optimizer.py').read().split("# Candidate rule descriptors")[0])
N=len(O);print('OOS27|n=%d'%N,flush=True)
H={}
for sym in sorted(set(x['sym'] for x in O)):
 xs=[x for x in O if x['sym']==sym];st=min(x['ts'] for x in xs)-pd.Timedelta(days=180);en=max(x['ts'] for x in xs)+pd.Timedelta(hours=1081)
 try:H[sym]=hourly(sym,int(st.timestamp()*1000),int(en.timestamp()*1000));print('H27|%s|%d'%(sym,len(H[sym])),flush=True)
 except Exception as e:H[sym]=pd.DataFrame();print('FAIL27|%s|%s'%(sym,str(e)[:100]),flush=True)
def firstbear(p,tf,start=6):
 z=bars(p,tf);b,s=psar_state(z)
 for j in range(start,len(p)):
  k=z.index.searchsorted(p.index[j],'right')-1
  if k>=0 and not b[k]:return j
 return len(p)-1
def atr(df,n=14):
 tr=pd.concat([df.high-df.low,(df.high-df.close.shift()).abs(),(df.low-df.close.shift()).abs()],axis=1).max(axis=1);return tr.ewm(alpha=1/n,adjust=False).mean()
def adx(df,n=14):
 h,l,c=df.high,df.low,df.close;up=h.diff();dn=-l.diff();tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1);a=tr.ewm(alpha=1/n,adjust=False).mean();pp=up.where((up>dn)&(up>0),0).ewm(alpha=1/n,adjust=False).mean();mm=dn.where((dn>up)&(dn>0),0).ewm(alpha=1/n,adjust=False).mean();P=100*pp/a.replace(0,np.nan);M=100*mm/a.replace(0,np.nan);X=100*(P-M).abs()/(P+M).replace(0,np.nan);return X.ewm(alpha=1/n,adjust=False).mean(),P,M
rows=[]
for ci,x in enumerate(O):
 full=H.get(x['sym'],pd.DataFrame());hist=full[full.index<=x['ts']];zw=bars(hist,168)
 if len(zw)<5:continue
 bw,sw=psar_state(zw);kw=zw.index.searchsorted(x['ts'],'right')-1
 if kw<0 or not bw[kw]:continue
 p=full[(full.index>=x['ts'])&(full.index<x['ts']+pd.Timedelta(hours=1081))]
 if len(p)<48:continue
 entry=float(x['entry']);j8=firstbear(p,8);j12=firstbear(p,12);j24=firstbear(p,24);px8=float(p.close.iloc[j8]);r8=px8/entry-1;r12=float(p.close.iloc[j12])/entry-1;r24=float(p.close.iloc[j24])/entry-1
 # features known exactly at first 8H bearish event
 pre=p.iloc[:j8+1];runhi=float(pre.high.max());mfe=runhi/entry-1;give=px8/runhi-1;age=j8/24;wd=(entry-float(sw[kw]))/entry if np.isfinite(sw[kw]) else 0
 rr=pre.close.pct_change();F=[r8,mfe,give,age,wd,float(rr.tail(24).std()),float(rr.tail(72).std())]
 for lag in [6,12,24,72]:F.append(float(px8/pre.close.iloc[max(0,len(pre)-1-lag)]-1))
 breadth=0
 for tf in [2,4,6,8,12,24]:
  z=bars(pre,tf);b,s=psar_state(z);k=len(z)-1;be=float(k>=0 and not b[k]);breadth+=be;F += [be,(px8-float(s[k]))/px8 if k>=0 and np.isfinite(s[k]) else 0]
 q=bars(pre,8);A=atr(q);X,P,M=adx(q);e20=q.close.ewm(span=20,adjust=False).mean();e50=q.close.ewm(span=50,adjust=False).mean();k=len(q)-1
 vv=lambda s: float(s.iloc[k]) if k>=0 and np.isfinite(s.iloc[k]) else 0
 F += [breadth,vv(X),vv(P)-vv(M),vv(A)/px8,(px8-vv(e20))/px8,(px8-vv(e50))/px8]
 rows.append([ci,x['sym'],x['ts'],r8,r12,r24,r12-r8,r24-r8]+F)
fn=['r8','mfe','give','age','wdist','vol24','vol72','mom6','mom12','mom24','mom72']
for tf in [2,4,6,8,12,24]:fn += ['bear%d'%tf,'psard%d'%tf]
fn += ['breadth','adx8','disp8','atr8','e20d','e50d']
D=pd.DataFrame(rows,columns=['ci','sym','ts','sell8','sell12','sell24','d12','d24']+fn).replace([np.inf,-np.inf],np.nan).fillna(0);print('ROWS27|%d|features=%d|d12med=%.4f|d24med=%.4f'%(len(D),len(fn),D.d12.median(),D.d24.median()),flush=True)
# Simple conditional rules first: choose 8/12/24 by weekly distance and current profit bins, calibrated OOS-style
D['wbin']=pd.qcut(D.wdist,4,labels=False,duplicates='drop');D['pbin']=pd.cut(D.r8,[-99,-.03,0,.03,.08,99],labels=False)
for col in ['wbin','pbin']:
 for z,g in D[D.ci>=430].groupby(col):
  vals=[]
  for h,c in [(8,'sell8'),(12,'sell12'),(24,'sell24')]:
   r=np.clip(g[c].values,-.95,None);vals.append((np.exp(np.mean(np.log1p(r)))-1,h))
  vals.sort(reverse=True);print('BIN27|%s=%s|n=%d|win=%dh|geo=%.4f'%(col,str(z),len(g),vals[0][1],vals[0][0]),flush=True)
# ML predicts incremental log reward of 12 and 24 vs 8, then action chosen only if predicted edge exceeds threshold selected on past calibration
for target,col in [('12','d12'),('24','d24')]:
 for mn in ['ET','HGB']:
  outs=[]
  for fi,(te,va,vb) in enumerate([(430,430,600),(600,600,760),(760,760,920),(920,920,N)],1):
   tr=D.ci<te;cal=D[(D.ci>=int(te*.75))&(D.ci<te)]
   m=ExtraTreesRegressor(n_estimators=500,max_depth=8,min_samples_leaf=12,n_jobs=-1,random_state=270+fi) if mn=='ET' else HistGradientBoostingRegressor(max_iter=250,max_leaf_nodes=12,learning_rate=.035,l2_regularization=4,random_state=270+fi)
   # target log-return difference for compounding objective
   y=np.log1p(np.clip(D.loc[tr,'sell'+target].values,-.95,None))-np.log1p(np.clip(D.loc[tr,'sell8'].values,-.95,None));m.fit(D.loc[tr,fn],y)
   best=(-1e9,0)
   for th in [0,.0025,.005,.01,.015,.02,.03]:
    if cal.empty:continue
    pred=m.predict(cal[fn]);use=pred>th;r=np.where(use,cal['sell'+target],cal.sell8);sc=np.mean(np.log1p(np.clip(r,-.95,None)))
    if sc>best[0]:best=(sc,th)
   q=D[(D.ci>=va)&(D.ci<vb)];pred=m.predict(q[fn]);use=pred>best[1];base=np.clip(q.sell8.values,-.95,None);gate=np.clip(np.where(use,q['sell'+target],q.sell8),-.95,None);bl=np.mean(np.log1p(base));gl=np.mean(np.log1p(gate));print('FOLD27|hold%s|%s|f=%d|thr=%.4f|n=%d|basegeo=%.4f|gategeo=%.4f|dlog=%.5f|use=%.3f|basep10=%.4f|gatep10=%.4f'%(target,mn,fi,best[1],len(q),np.exp(bl)-1,np.exp(gl)-1,gl-bl,use.mean() if len(use) else 0,np.quantile(base,.1) if len(base) else 0,np.quantile(gate,.1) if len(gate) else 0),flush=True);outs.append((gl-bl,np.exp(bl)-1,np.exp(gl)-1,use.mean() if len(use) else 0))
  a=np.array(outs);print('AGG27|hold%s|%s|wins=%d/4|dlogmed=%.5f|basegeo=%.4f|gategeo=%.4f|use=%.3f'%(target,mn,int((a[:,0]>0).sum()),np.median(a[:,0]),np.median(a[:,1]),np.median(a[:,2]),np.median(a[:,3])),flush=True)
print('DONE27',flush=True)