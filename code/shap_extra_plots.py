# -*- coding: utf-8 -*-
"""
Extra SHAP visualisations requested by the reviewer (strengthen Explainability):
  * SHAP Summary (beeswarm)      -> shap_summary.png
  * SHAP Dependence (top driver) -> shap_dependence_mtbf.png  + lagged-failures
  * SHAP Force plot (one month)  -> shap_force.png

Same surrogate as shap_run.py: XGBoost on monthly one-step-ahead features,
target = next-month failure count, exact TreeSHAP. REAL values only.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "Table3_Daily_Maintenance_Data.csv")

daily = pd.read_csv(CSV); daily.columns=[c.strip() for c in daily.columns]
MONTHS={m:i+1 for i,m in enumerate(["January","February","March","April","May","June",
        "July","August","September","October","November","December"])}
daily["month_num"]=daily["Month"].map(MONTHS)
df=(daily.groupby(["Year","month_num"],as_index=False)
        .agg(Failures=("Failures","sum"),Downtime=("Downtime (min)","sum"),
             Operating=("Operating (min)","sum")))
df=df.sort_values(["Year","month_num"]).reset_index(drop=True)
df["MTBF"]=df["Operating"]/df["Failures"].replace(0,np.nan)
df["MTTR"]=df["Downtime"]/df["Failures"].replace(0,np.nan)
df["MTTR"]=df["MTTR"].fillna(0.0); df["MTBF"]=df["MTBF"].fillna(df["Operating"])

feat=pd.DataFrame()
feat["Lagged failures"] = df["Failures"].shift(1)
feat["Operating hours"] = df["Operating"].shift(1)
feat["MTBF"]            = df["MTBF"].shift(1)
feat["MTTR"]            = df["MTTR"].shift(1)
feat["Downtime"]        = df["Downtime"].shift(1)
feat["Month"]           = df["month_num"]
feat["Year index"]      = df["Year"]-df["Year"].min()
y=df["Failures"]
ok=feat.notna().all(axis=1)&y.notna()
X=feat[ok].reset_index(drop=True); y=y[ok].reset_index(drop=True)
yr=df["Year"][ok].reset_index(drop=True); mo=df["month_num"][ok].reset_index(drop=True)
print("samples",len(X))

model=XGBRegressor(n_estimators=300,max_depth=3,learning_rate=0.05,
                   subsample=0.9,colsample_bytree=1.0,random_state=0).fit(X,y)
expl=shap.TreeExplainer(model)
sv=expl(X)

# ---- 1) SHAP Summary (beeswarm) ----
plt.figure()
shap.summary_plot(sv, X, show=False, max_display=len(X.columns))
plt.title("SHAP summary (impact on predicted next-month failures)", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(HERE,"shap_summary.png"),dpi=200,bbox_inches="tight"); plt.close()

# ---- 2) SHAP Dependence (top driver = MTBF, auto-coloured by strongest interaction) ----
plt.figure()
shap.dependence_plot("MTBF", sv.values, X, show=False)
plt.tight_layout(); plt.savefig(os.path.join(HERE,"shap_dependence_mtbf.png"),dpi=200,bbox_inches="tight"); plt.close()

plt.figure()
shap.dependence_plot("Lagged failures", sv.values, X, show=False)
plt.tight_layout(); plt.savefig(os.path.join(HERE,"shap_dependence_lagfail.png"),dpi=200,bbox_inches="tight"); plt.close()

# ---- 3) SHAP Force plot for one representative month ----
# pick the instance with the largest absolute deviation of prediction from the mean
pred=model.predict(X); base=float(expl.expected_value)
i=int(np.argmax(np.abs(pred-base)))
print(f"force instance idx={i}  year={int(yr[i])} month={int(mo[i])}  pred={pred[i]:.2f}  base={base:.2f}")
disp=X.iloc[i].round(0).astype(int)   # clean integer display values (SHAP values unchanged)
shap.force_plot(base, sv.values[i], disp, matplotlib=True, show=False,
                text_rotation=12, contribution_threshold=0.05)
fig=plt.gcf(); fig.set_size_inches(12,3.2)
plt.tight_layout(); plt.savefig(os.path.join(HERE,"shap_force.png"),dpi=200,bbox_inches="tight"); plt.close()

print("Saved: shap_summary.png, shap_dependence_mtbf.png, shap_dependence_lagfail.png, shap_force.png")
print(f"base value (expected next-month failures) = {base:.3f}")
