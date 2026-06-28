# -*- coding: utf-8 -*-
"""
EXTERNAL VALIDATION of the ARW-PI adaptive-gating mechanism on an INDEPENDENT public
benchmark: the AI4I 2020 Predictive Maintenance dataset (UCI id 601, 10,000 rows).

Goal: test whether the paper's central modelling claim — that an operating-condition-
novelty gate blending a data model with a robust anchor gives BEST-OF-BOTH robustness
under operating-regime shift (Proposition 1) — transfers beyond the single factory.

Task: predict 'Machine failure' (binary, ~3.4% positives) from the process variables.
Two regimes (mirroring the paper):
  STATIONARY : stratified random 70/30 split (in-distribution).
  SHIFT      : train on low tool-wear (<= 60th pct), test on high tool-wear (> 60th pct)
               -> an operating-condition shift, exactly what the gate is designed for.
Models: data model (XGBoost), robust anchor (L2 logistic regression), and the adaptive
hybrid p = alpha*p_anchor + (1-alpha)*p_data with alpha = sigma(s*(d_oc - tau)), where
d_oc is the standardized novelty of each test point vs the training operating distribution.
Metrics: ROC-AUC and PR-AUC (average precision; appropriate for the class imbalance),
mean over 5 seeds. NO fabricated numbers.
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

ds = fetch_ucirepo(id=601)
X = ds.data.features.copy(); y = ds.data.targets["Machine failure"].astype(int).values
X["Type"] = X["Type"].map({"L":0,"M":1,"H":2}).astype(float)
COLS = ["Type","Air temperature","Process temperature","Rotational speed","Torque","Tool wear"]
CONT = ["Air temperature","Process temperature","Rotational speed","Torque","Tool wear"]
X = X[COLS].astype(float)
print(f"AI4I 2020: {len(X)} rows, {y.mean()*100:.2f}% failures")

def sigmoid(z): return 1/(1+np.exp(-z))

def evaluate(tr, te, seeds=(0,1,2,3,4), s=3.0, tau=2.0):
    Xtr, Xte = X.iloc[tr], X.iloc[te]; ytr, yte = y[tr], y[te]
    sc = StandardScaler().fit(Xtr[CONT])
    Ztr, Zte = sc.transform(Xtr[CONT]), sc.transform(Xte[CONT])
    # operating-condition novelty: standardized Euclidean distance from train centroid,
    # normalized by the train-typical distance
    d_tr = np.sqrt((Ztr**2).sum(1)); d_te = np.sqrt((Zte**2).sum(1))
    d_oc = d_te / (np.median(d_tr)+1e-9)
    alpha = sigmoid(s*(d_oc - tau))                       # ->1 (anchor) when novel
    # robust anchor: L2 logistic regression on standardized features
    anc = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)
    anc.fit(Ztr, ytr); p_anc = anc.predict_proba(Zte)[:,1]
    res = {"data":[], "anchor":[], "static":[], "ARW-PI":[]}
    for seed in seeds:
        dm = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                           subsample=0.9, n_jobs=1, random_state=seed,
                           eval_metric="logloss",
                           scale_pos_weight=(len(ytr)-ytr.sum())/max(ytr.sum(),1))
        dm.fit(Xtr.values, ytr); p_dat = dm.predict_proba(Xte.values)[:,1]
        p_hyb = alpha*p_anc + (1-alpha)*p_dat            # adaptive
        p_sta = 0.5*p_anc + 0.5*p_dat                    # static blend (gate removed)
        for name,p in [("data",p_dat),("anchor",p_anc),("static",p_sta),("ARW-PI",p_hyb)]:
            res[name].append((roc_auc_score(yte,p), average_precision_score(yte,p)))
    return {k:np.mean(v,0) for k,v in res.items()}, alpha.mean()

rng = np.random.default_rng(0)
# STATIONARY: stratified random 70/30
idx = np.arange(len(X)); rng.shuffle(idx)
pos, neg = idx[y[idx]==1], idx[y[idx]==0]
def split7030(a): c=int(0.7*len(a)); return a[:c], a[c:]
trp,tep = split7030(pos); trn,ten = split7030(neg)
tr_s, te_s = np.r_[trp,trn], np.r_[tep,ten]
# SHIFT: train low tool-wear, test high tool-wear
tw = X["Tool wear"].values; thr = np.quantile(tw, 0.60)
tr_h, te_h = np.where(tw<=thr)[0], np.where(tw>thr)[0]

for label,(tr,te) in {"STATIONARY (random 70/30)":(tr_s,te_s),
                      "SHIFT (train tool-wear<=60th pct, test >60th pct)":(tr_h,te_h)}.items():
    r, amean = evaluate(tr,te)
    print(f"\n================ {label} ================")
    print(f"  (test failures: {y[te].mean()*100:.2f}% | mean gate alpha: {amean:.2f})")
    print(f"  {'model':10s} {'ROC-AUC':>9} {'PR-AUC':>9}")
    for k in ["data","anchor","static","ARW-PI"]:
        print(f"  {k:10s} {r[k][0]:9.3f} {r[k][1]:9.3f}")
print("\nDONE.")
