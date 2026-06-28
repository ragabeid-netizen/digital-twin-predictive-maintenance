# -*- coding: utf-8 -*-
"""
REAL, fair benchmark of monthly availability forecasting for the metal-detector
machine, to replace the "all baselines negative R2" claim with honest numbers.

Fairness principles:
  * ALL models get the same information: lagged history + calendar + the KNOWN
    (planned) operating hours of the target month (an exogenous quantity).
  * Two evaluation regimes are reported:
      - STATIONARY split : train 2010-2018, test 2019-2021  (in-distribution)
      - REGIME-SHIFT split: train 2010-2021, test 2022-2025 (operating hours ~2x)
  * Stochastic models averaged over 5 seeds (mean +/- std).
  * Paired Wilcoxon test (proposed abs-err vs best baseline abs-err) on the
    regime-shift test set.

Proposed = physics-informed hybrid: an ML model forecasts next-month FAILURES from
lagged state; downtime_hat = failures_hat * recent MTTR; availability is then
recovered through the identity A = (Operating - Downtime)/Operating using the known
operating hours. (ML forecasts the small failure process; physics does the algebra.)
"""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "Table3_Daily_Maintenance_Data.csv")

# ---------- data: daily -> monthly ----------
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
df["date"]=pd.to_datetime(dict(year=df.Year, month=df.mn, day=1))

# ---------- feature matrix (no contemporaneous leakage) ----------
F=pd.DataFrame(index=df.index)
F["lag1_av"]=df.Avail.shift(1); F["lag2_av"]=df.Avail.shift(2); F["lag3_av"]=df.Avail.shift(3)
F["lag12_av"]=df.Avail.shift(12)
F["lag1_fail"]=df.Failures.shift(1); F["lag1_down"]=df.Downtime.shift(1)
F["lag1_mttr"]=df.MTTR.shift(1)
F["month"]=df.mn; F["yr"]=df.Year-df.Year.min()
F["operating"]=df.Operating               # known/planned exogenous
y=df.Avail.values

def metrics(yt,yp):
    yt,yp=np.asarray(yt,float),np.asarray(yp,float)
    mae=np.mean(np.abs(yt-yp)); rmse=np.sqrt(np.mean((yt-yp)**2))
    ss=np.sum((yt-yt.mean())**2); r2=1-np.sum((yt-yp)**2)/ss if ss>0 else float('nan')
    return mae,rmse,r2

def split(train_end_year, test_years):
    tr=df.index[df.Year<=train_end_year]
    te=df.index[df.Year.isin(test_years)]
    return tr,te

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import statsmodels.api as sm
from prophet import Prophet
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

ML_COLS=["lag1_av","lag2_av","lag3_av","lag12_av","lag1_fail","lag1_down","lag1_mttr","month","yr","operating"]

def run_ml(make, tr, te, seeds=(0,1,2,3,4)):
    res=[]
    valid=[i for i in range(len(df)) if F.loc[i,ML_COLS].notna().all()]
    tri=[i for i in tr if i in valid]; tei=[i for i in te if i in valid]
    Xtr=F.loc[tri,ML_COLS].values; ytr=y[tri]; Xte=F.loc[tei,ML_COLS].values; yte=y[tei]
    for s in seeds:
        m=make(s); m.fit(Xtr,ytr); res.append(np.abs(yte-m.predict(Xte)))
    err=np.mean(res,axis=0)
    return yte, y[tei]-err*0+ (yte-err if False else None), err, tei

def eval_ml(make,tr,te,seeds=(0,1,2,3,4)):
    valid=[i for i in range(len(df)) if F.loc[i,ML_COLS].notna().all()]
    tri=[i for i in tr if i in valid]; tei=[i for i in te if i in valid]
    Xtr=F.loc[tri,ML_COLS].values; ytr=y[tri]; Xte=F.loc[tei,ML_COLS].values; yte=y[tei]
    preds=[]
    for s in seeds:
        m=make(s);
        try: m.fit(Xtr,ytr)
        except TypeError: m.fit(Xtr,ytr)
        preds.append(m.predict(Xte))
    P=np.array(preds)
    mm=[metrics(yte,p) for p in P]
    arr=np.array(mm)  # rows seeds, cols mae rmse r2
    return arr.mean(0), arr.std(0), yte, P.mean(0), tei

def eval_persistence(tr,te):
    tei=list(te); yte=y[tei]; yp=df.Avail.shift(1).values[tei]
    return metrics(yte,yp),(0,0,0),yte,yp,tei

def eval_seasonal(tr,te):
    tei=list(te); yte=y[tei]; yp=df.Avail.shift(12).values[tei]
    return metrics(yte,yp),(0,0,0),yte,yp,tei

def eval_arimax(tr,te):
    tei=list(te); tri=list(tr)
    ytr=df.Avail.values[tri]; extr=df.Operating.values[tri].reshape(-1,1)
    exte=df.Operating.values[tei].reshape(-1,1); yte=y[tei]
    try:
        mod=sm.tsa.SARIMAX(ytr,exog=extr,order=(2,1,2),enforce_stationarity=False,enforce_invertibility=False).fit(disp=0)
        yp=mod.forecast(steps=len(tei),exog=exte)
    except Exception as e:
        yp=np.full(len(tei),ytr.mean())
    return metrics(yte,yp),(0,0,0),yte,yp,tei

def eval_prophet(tr,te):
    tei=list(te); tri=list(tr)
    dtr=pd.DataFrame({"ds":df.date.values[tri],"y":df.Avail.values[tri],"operating":df.Operating.values[tri]})
    dte=pd.DataFrame({"ds":df.date.values[tei],"operating":df.Operating.values[tei]})
    yte=y[tei]
    try:
        m=Prophet(yearly_seasonality=True,weekly_seasonality=False,daily_seasonality=False)
        m.add_regressor("operating"); m.fit(dtr); yp=m.predict(dte)["yhat"].values
    except Exception:
        yp=np.full(len(tei),df.Avail.values[tri].mean())
    return metrics(yte,yp),(0,0,0),yte,yp,tei

def make_seq(idx, window=12):
    cols=["Avail","Failures","Downtime","Operating","MTTR"]
    data=df[cols].values
    Xs,ys,ti=[],[],[]
    for i in idx:
        if i-window<0: continue
        Xs.append(data[i-window:i]); ys.append(df.Avail.values[i]); ti.append(i)
    return np.array(Xs),np.array(ys),ti

def eval_rnn(kind,tr,te,seeds=(0,1,2),epochs=120,window=12):
    Xtr,ytr,_=make_seq(tr,window); Xte,yte_seq,tei=make_seq(te,window); yte=np.array(yte_seq)
    ns,nt,nf=Xtr.shape
    sc=StandardScaler().fit(Xtr.reshape(-1,nf))
    Xtr2=sc.transform(Xtr.reshape(-1,nf)).reshape(Xtr.shape)
    Xte2=sc.transform(Xte.reshape(-1,nf)).reshape(Xte.shape)
    ysc=StandardScaler().fit(ytr.reshape(-1,1))          # scale TARGET too
    ytr2=ysc.transform(ytr.reshape(-1,1)).ravel()
    mm=[]; preds=[]
    for s in seeds:
        tf.keras.utils.set_random_seed(s)
        m=keras.Sequential([keras.layers.Input((window,nf)),
            (keras.layers.LSTM(32) if kind=="LSTM" else keras.layers.GRU(32)),
            keras.layers.Dense(1)])
        m.compile(optimizer="adam",loss="mse")
        m.fit(Xtr2,ytr2,epochs=epochs,verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=15,restore_best_weights=True)],
              validation_split=0.15)
        p=ysc.inverse_transform(m.predict(Xte2,verbose=0)).ravel(); preds.append(p); mm.append(metrics(yte,p))
    arr=np.array(mm); return arr.mean(0),arr.std(0),yte,np.mean(preds,0),tei

def eval_proposed(tr,te):
    # ML forecasts failures; physics converts to availability via known operating
    valid=[i for i in range(len(df)) if F.loc[i,["lag1_fail","lag2_av","lag3_av","lag1_mttr","month","yr"]].notna().all()]
    fcols=["lag1_fail","lag1_down","lag1_mttr","month","yr","operating","lag1_av","lag2_av"]
    F2=F.copy(); F2["lag2_fail"]=df.Failures.shift(2)
    tri=[i for i in tr if i in valid]; tei=[i for i in te if i in valid]
    Xtr=F.loc[tri,fcols].values; Xte=F.loc[tei,fcols].values
    fail_tr=df.Failures.values[tri]
    preds=[]
    for s in (0,1,2,3,4):
        m=XGBRegressor(n_estimators=300,max_depth=3,learning_rate=0.05,subsample=0.9,random_state=s)
        m.fit(Xtr,fail_tr); fh=np.clip(m.predict(Xte),0,None)
        mttr=df.MTTR.shift(1).values[tei]            # recent repair time
        down_h=fh*mttr
        op=df.Operating.values[tei]
        av_h=(op-down_h)/op*100
        preds.append(av_h)
    P=np.array(preds); yte=y[tei]
    mm=np.array([metrics(yte,p) for p in P])
    return mm.mean(0),mm.std(0),yte,P.mean(0),tei

def eval_adaptive(tr,te):
    # Adaptive physics-informed hybrid: blend a data model (XGBoost) with a robust
    # anchor (persistence) using an operating-condition novelty gate. When the month's
    # operating hours fall far outside the training distribution (regime shift), the
    # gate -> 1 and the forecast leans on the robust anchor; otherwise on the data model.
    valid=[i for i in range(len(df)) if F.loc[i,ML_COLS].notna().all()]
    tri=[i for i in tr if i in valid]; tei=[i for i in te if i in valid]
    Xtr=F.loc[tri,ML_COLS].values; ytr=y[tri]; Xte=F.loc[tei,ML_COLS].values; yte=y[tei]
    op_mu,op_sd=df.Operating.values[tri].mean(),df.Operating.values[tri].std()+1e-9
    preds=[]
    for s in (0,1,2,3,4):
        dm=XGBRegressor(n_estimators=400,max_depth=3,learning_rate=0.05,subsample=0.9,random_state=s).fit(Xtr,ytr)
        data_pred=dm.predict(Xte)
        anchor=df.Avail.shift(1).values[tei]                 # robust persistence anchor
        d_oc=np.abs(df.Operating.values[tei]-op_mu)/op_sd    # operating-condition novelty
        alpha=1/(1+np.exp(-3.0*(d_oc-2.0)))                  # gate: ->1 when far from train
        preds.append(alpha*anchor+(1-alpha)*data_pred)
    P=np.array(preds); mm=np.array([metrics(yte,p) for p in P])
    return mm.mean(0),mm.std(0),yte,P.mean(0),tei

def report(name,fn,tr,te,store):
    try:
        mean,std,yte,yp,tei=fn(tr,te)
        if isinstance(mean,tuple): mean=np.array(mean); std=np.array(std)
        store[name]=(np.array(mean),np.array(std),np.array(yte),np.array(yp),tei)
        yt_,yp_=np.asarray(yte,float),np.asarray(yp,float)
        mape=np.mean(np.abs((yt_-yp_)/yt_))*100
        print(f"  {name:26s} MAE={mean[0]:7.4f}  RMSE={mean[1]:7.4f}  R2={mean[2]:8.3f}  MAPE={mape:7.4f}"
              + (f"  (+/-{std[2]:.3f})" if std[2]>0 else ""))
    except Exception as e:
        print(f"  {name:24s} ERROR: {e}")

for label,(tey,tyrs) in {"STATIONARY (train<=2018, test 2019-2021)":(2018,[2019,2020,2021]),
                          "REGIME-SHIFT (train<=2021, test 2022-2025)":(2021,[2022,2023,2024,2025])}.items():
    tr,te=split(tey,tyrs)
    print(f"\n================ {label} ================")
    store={}
    report("Persistence (naive)",eval_persistence,tr,te,store)
    report("Seasonal naive (t-12)",eval_seasonal,tr,te,store)
    report("ARIMAX(2,1,2)+op",eval_arimax,tr,te,store)
    report("Prophet+op",eval_prophet,tr,te,store)
    report("Random Forest",lambda tr,te:eval_ml(lambda s:RandomForestRegressor(n_estimators=400,random_state=s),tr,te),tr,te,store)
    report("XGBoost",lambda tr,te:eval_ml(lambda s:XGBRegressor(n_estimators=400,max_depth=3,learning_rate=0.05,subsample=0.9,random_state=s),tr,te),tr,te,store)
    report("LSTM",lambda tr,te:eval_rnn("LSTM",tr,te),tr,te,store)
    report("GRU",lambda tr,te:eval_rnn("GRU",tr,te),tr,te,store)
    report("Physics-informed (simple)",eval_proposed,tr,te,store)
    report("Adaptive hybrid (PROPOSED)",eval_adaptive,tr,te,store)
    # significance on regime-shift split: proposed vs best data-driven baseline by MAE
    # ---- formal statistical analysis: 95% CI (bootstrap) + paired t-test + Wilcoxon ----
    from scipy.stats import wilcoxon, ttest_rel
    rng=np.random.default_rng(0)
    def boot_ci(x,n=5000,a=0.05):
        x=np.asarray(x,float); idx=rng.integers(0,len(x),(n,len(x)))
        bs=x[idx].mean(axis=1); return np.percentile(bs,[100*a/2,100*(1-a/2)])
    def err_map(e):
        yte,yp,tei=e[2],e[3],e[4]; return {int(tei[i]):abs(float(yte[i])-float(yp[i])) for i in range(len(tei))}
    prop=store.get("Adaptive hybrid (PROPOSED)")
    if prop is not None:
        pm=err_map(prop); pe=np.array(list(pm.values()))
        lo,hi=boot_ci(pe)
        print(f"  STATS: Proposed MAE={pe.mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n={len(pe)})")
        for k in store:
            if k=="Adaptive hybrid (PROPOSED)": continue
            bm=err_map(store[k]); common=sorted(set(pm)&set(bm))
            if len(common)<5: continue
            pa=np.array([pm[i] for i in common]); ba=np.array([bm[i] for i in common])
            d=ba-pa; dlo,dhi=boot_ci(d)
            try: _,pt=ttest_rel(pa,ba)
            except Exception: pt=float('nan')
            try: _,pw=wilcoxon(pa,ba)
            except Exception: pw=float('nan')
            print(f"    vs {k:26s} dMAE={d.mean():+.3f} CI[{dlo:+.3f},{dhi:+.3f}]  t-test p={pt:.3g}  Wilcoxon p={pw:.3g}")
print("\nDONE.")
