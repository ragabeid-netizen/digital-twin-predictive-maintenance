# -*- coding: utf-8 -*-
"""
Ablation study, sensitivity analysis, bootstrap confidence intervals and effect
sizes for the Adaptive Reliability-Weighted (ARW-PI) availability estimator,
to satisfy Q1 reviewer comment 2. Reuses the exact monthly pipeline of
benchmark_run.py. ALL numbers are computed from the real data -- nothing is
hand-set. Headline scenario = the 2022-2025 regime-shift hold-out
(train <= 2021), with the stationary split (train <= 2018, test 2019-2021)
reported alongside for the ablation.
"""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "Table3_Daily_Maintenance_Data.csv")

# ---------- data: daily -> monthly (identical to benchmark_run.py) ----------
d = pd.read_csv(CSV); d.columns=[c.strip() for c in d.columns]
M={m:i+1 for i,m in enumerate(["January","February","March","April","May","June",
   "July","August","September","October","November","December"])}
d["mn"]=d["Month"].map(M)
df=(d.groupby(["Year","mn"],as_index=False)
      .agg(Failures=("Failures","sum"),Downtime=("Downtime (min)","sum"),
           Operating=("Operating (min)","sum")))
df=df.sort_values(["Year","mn"]).reset_index(drop=True)
df["MTTR"]=(df["Downtime"]/df["Failures"].replace(0,np.nan)).fillna(0)
df["Avail"]=(df["Operating"]-df["Downtime"])/df["Operating"]*100

F=pd.DataFrame(index=df.index)
F["lag1_av"]=df.Avail.shift(1); F["lag2_av"]=df.Avail.shift(2); F["lag3_av"]=df.Avail.shift(3)
F["lag12_av"]=df.Avail.shift(12)
F["lag1_fail"]=df.Failures.shift(1); F["lag1_down"]=df.Downtime.shift(1)
F["lag1_mttr"]=df.MTTR.shift(1)
F["month"]=df.mn; F["yr"]=df.Year-df.Year.min()
F["operating"]=df.Operating
y=df.Avail.values
ML=["lag1_av","lag2_av","lag3_av","lag12_av","lag1_fail","lag1_down","lag1_mttr","month","yr","operating"]

from xgboost import XGBRegressor
from scipy.stats import wilcoxon, ttest_rel

def metrics(yt,yp):
    yt,yp=np.asarray(yt,float),np.asarray(yp,float)
    mae=np.mean(np.abs(yt-yp)); rmse=np.sqrt(np.mean((yt-yp)**2))
    ss=np.sum((yt-yt.mean())**2); r2=1-np.sum((yt-yp)**2)/ss if ss>0 else float('nan')
    mape=np.mean(np.abs((yt-yp)/yt))*100
    return mae,rmse,r2,mape

def split(train_end,test_years):
    valid=[i for i in range(len(df)) if F.loc[i,ML].notna().all()]
    tr=[i for i in df.index[df.Year<=train_end] if i in valid]
    te=[i for i in df.index[df.Year.isin(test_years)] if i in valid]
    return tr,te

def gate_alpha(tri,tei,s,tau):
    op_mu,op_sd=df.Operating.values[tri].mean(),df.Operating.values[tri].std()+1e-9
    d_oc=np.abs(df.Operating.values[tei]-op_mu)/op_sd
    return 1/(1+np.exp(-s*(d_oc-tau)))

def predict(tri,tei,mode,seed,s=3.0,tau=2.0,alpha_fixed=0.5):
    """mode: 'data','anchor','static','adaptive'. Returns per-point predictions."""
    Xtr=F.loc[tri,ML].values; ytr=y[tri]; Xte=F.loc[tei,ML].values
    anchor=df.Avail.shift(1).values[tei]
    if mode=="anchor":
        return anchor
    dm=XGBRegressor(n_estimators=400,max_depth=3,learning_rate=0.05,subsample=0.9,
                    random_state=seed,n_jobs=1,verbosity=0).fit(Xtr,ytr)
    data_pred=dm.predict(Xte)
    if mode=="data":   return data_pred
    if mode=="static": a=np.full(len(tei),alpha_fixed)
    else:              a=gate_alpha(tri,tei,s,tau)          # adaptive
    return a*anchor+(1-a)*data_pred

def evaluate(tri,tei,mode,seeds=(0,1,2,3,4),**kw):
    yte=y[tei]; preds=[predict(tri,tei,mode,s,**kw) for s in seeds]
    mm=np.array([metrics(yte,p) for p in preds])
    return mm.mean(0),mm.std(0),np.mean(preds,0)

tri_s,tei_s=split(2018,[2019,2020,2021])      # stationary
tri_r,tei_r=split(2021,[2022,2023,2024,2025]) # regime shift

def fmt(m,s): return f"MAE={m[0]:.3f}±{s[0]:.3f} RMSE={m[1]:.3f}±{s[1]:.3f} R2={m[2]:.3f} MAPE={m[3]:.3f}"

import sys
def log(*a): print(*a); sys.stdout.flush()
log("pipeline ready; stationary n_test=%d, regime-shift n_test=%d"%(len(tei_s),len(tei_r)))
print("="*70)
print("ABLATION STUDY  (component contribution)")
print("="*70)
abl={}
for mode,label in [("data","Data model only (XGBoost), no anchor/gate  [alpha=0]"),
                   ("anchor","Robust anchor only (persistence)          [alpha=1]"),
                   ("static","Static blend (no adaptive gate)           [alpha=0.5]"),
                   ("adaptive","Full ARW-PI (adaptive gate)              [proposed]")]:
    ms_s=evaluate(tri_s,tei_s,mode); ms_r=evaluate(tri_r,tei_r,mode)
    abl[mode]=(ms_s,ms_r)
    print(f"\n{label}")
    print(f"   STATIONARY  : {fmt(ms_s[0],ms_s[1])}")
    print(f"   REGIME-SHIFT: {fmt(ms_r[0],ms_r[1])}")

print("\n"+"="*70)
print("SENSITIVITY ANALYSIS  (gate sharpness s, threshold tau) -- regime shift MAE")
print("="*70)
hdr="s\\tau"
print(f"{hdr:>8}"+"".join(f"{t:>10}" for t in [1.0,2.0,3.0]))
for s in [1.0,3.0,5.0,10.0]:
    row=f"{s:>8.1f}"
    for tau in [1.0,2.0,3.0]:
        m,_,_=evaluate(tri_r,tei_r,"adaptive",s=s,tau=tau)
        row+=f"{m[0]:>10.3f}"
    print(row)

print("\n"+"="*70)
print("CONFIDENCE INTERVALS + EFFECT SIZE  (regime-shift hold-out)")
print("="*70)
yte=y[tei_r]
_,_,prop=evaluate(tri_r,tei_r,"adaptive")
_,_,base=evaluate(tri_r,tei_r,"data")     # best learned baseline = XGBoost
pe=np.abs(yte-prop); be=np.abs(yte-base)
rng=np.random.default_rng(0)
def boot_ci(x,n=10000,a=0.05):
    idx=rng.integers(0,len(x),(n,len(x))); bs=x[idx].mean(1)
    return np.percentile(bs,[100*a/2,100*(1-a/2)])
lo,hi=boot_ci(pe)
print(f"Proposed MAE = {pe.mean():.3f}  95% bootstrap CI [{lo:.3f}, {hi:.3f}]  (n={len(pe)})")
lo2,hi2=boot_ci(be); print(f"XGBoost  MAE = {be.mean():.3f}  95% bootstrap CI [{lo2:.3f}, {hi2:.3f}]")
diff=be-pe
dlo,dhi=boot_ci(diff)
print(f"Paired MAE improvement (XGBoost - Proposed) = {diff.mean():.3f}  95% CI [{dlo:.3f}, {dhi:.3f}]")
w_stat,w_p=wilcoxon(pe,be); t_stat,t_p=ttest_rel(be,pe)
print(f"Wilcoxon signed-rank p = {w_p:.3g}   paired t-test p = {t_p:.3g}")
# effect sizes
dz=diff.mean()/(diff.std(ddof=1)+1e-12)                    # Cohen's dz (paired)
# Cliff's delta on paired errors (prob be>pe minus prob be<pe)
gt=np.sum(be[:,None]>pe[None,:]); lt=np.sum(be[:,None]<pe[None,:])
cliff=(gt-lt)/(len(pe)*len(be))
# rank-biserial from Wilcoxon
n=len(diff); dnz=diff[diff!=0];
from scipy.stats import rankdata
r=rankdata(np.abs(dnz)); Rpos=r[dnz>0].sum(); Rneg=r[dnz<0].sum()
rb=(Rpos-Rneg)/(Rpos+Rneg)
print(f"Effect size: Cohen's dz = {dz:.3f} | Cliff's delta = {cliff:.3f} | rank-biserial r = {rb:.3f}")
print("\nDONE.")
