"""Phase31: validate a 0-100 entry-quality score for Daily PSAR cross, frozen Phase29 exit. Score is percentile rank of causal ET prediction trained only on past. Test monotonic deciles/quintiles and same-day cross ranking."""
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
# Reuse Phase30 construction but stop before its ML section
src=open('sar_phase30_entry_filters.py').read(); src=src.split("# Walk-forward ML ranking:")[0]; exec(compile(src,'phase30_build','exec'))
print('START31|rows=%d|features=%d'%(len(D),len(fn)),flush=True)
folds=[(430,600),(600,760),(760,920),(920,N)]; scored=[]
for fi,(a,b) in enumerate(folds,1):
 tr=D[D.ci<a].copy();q=D[(D.ci>=a)&(D.ci<b)].copy()
 y=np.log1p(np.clip(tr.ret.values,-.95,None));m=ExtraTreesRegressor(n_estimators=800,max_depth=8,min_samples_leaf=15,n_jobs=-1,random_state=310+fi);m.fit(tr[fn],y)
 pt=m.predict(tr[fn]);pq=m.predict(q[fn]);# percentile against historical training prediction distribution only
 s=np.searchsorted(np.sort(pt),pq,side='right')/len(pt)*100
 q['score']=np.clip(s,0,100);q['fold']=fi;scored.append(q);print('SCOREFOLD31|f=%d|n=%d|scoreMed=%.1f'%(fi,len(q),np.median(s)),flush=True)
S=pd.concat(scored,ignore_index=True);print('ROWS31|%d'%len(S),flush=True)
# fixed score bands: 0-20...80-100 and broad operational bands
for edges,name in [([0,20,40,60,80,101],'quint'),([0,40,60,75,101],'oper')]:
 S['band']=pd.cut(S.score,edges,right=False,labels=False)
 for z,g in S.groupby('band'):
  r=np.clip(g.ret.values,-.95,None);print('BAND31|%s|b=%d|n=%d|geo=%.4f|med=%.4f|win=%.3f|p10=%.4f'%(name,z,len(g),np.exp(np.mean(np.log1p(r)))-1,np.median(r),np.mean(r>0),np.quantile(r,.1)),flush=True)
# monotonicity by fold: correlation score vs future log return, and top vs bottom
for fi,g in S.groupby('fold'):
 corr=pd.Series(g.score).corr(pd.Series(np.log1p(np.clip(g.ret,-.95,None))),method='spearman');lo=g[g.score<40];hi=g[g.score>=75];geo=lambda x:np.exp(np.mean(np.log1p(np.clip(x,-.95,None))))-1 if len(x) else np.nan
 print('MONO31|f=%d|rho=%.3f|loN=%d|loGeo=%.4f|hiN=%d|hiGeo=%.4f'%(fi,corr,len(lo),geo(lo.ret),len(hi),geo(hi.ret)),flush=True)
# same-day ranking: when >=2 Daily crosses on same UTC day, does highest score outperform peers?
S['day']=pd.to_datetime(S.ts,utc=True).dt.floor('D');ds=[]
for day,g in S.groupby('day'):
 if len(g)<2:continue
 top=g.loc[g.score.idxmax()];others=g.drop(top.name);ds.append((float(top.ret),float(others.ret.mean()),len(g),float(top.score)))
A=pd.DataFrame(ds,columns=['top','other','n','score']);d=np.log1p(np.clip(A.top,-.95,None))-np.log1p(np.clip(A.other,-.95,None));print('SAMEDAY31|days=%d|topGeo=%.4f|otherGeo=%.4f|wins=%.3f|dlog=%.5f'%(len(A),np.exp(np.mean(np.log1p(np.clip(A.top,-.95,None))))-1,np.exp(np.mean(np.log1p(np.clip(A.other,-.95,None))))-1,np.mean(A.top>A.other),d.mean()),flush=True)
# bootstrap high score >=75 vs all and same-day top edge
rng=np.random.default_rng(3101);base=np.log1p(np.clip(S.ret.values,-.95,None));hi=np.log1p(np.clip(S[S.score>=75].ret.values,-.95,None));boots=[]
for _ in range(5000):boots.append(np.mean(rng.choice(hi,len(hi),True))-np.mean(rng.choice(base,len(base),True)))
boots=np.array(boots);print('BOOT31|high75_vs_all|edge=%.5f|pgt0=%.3f|ci95=[%.5f,%.5f]'%(hi.mean()-base.mean(),np.mean(boots>0),*np.quantile(boots,[.025,.975])),flush=True)
if len(d):
 bb=np.array([np.mean(rng.choice(d,len(d),True)) for _ in range(5000)]);print('BOOT31|sameday_top|edge=%.5f|pgt0=%.3f|ci95=[%.5f,%.5f]'%(d.mean(),np.mean(bb>0),*np.quantile(bb,[.025,.975])),flush=True)
print('DONE31',flush=True)