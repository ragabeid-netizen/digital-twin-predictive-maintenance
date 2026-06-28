# -*- coding: utf-8 -*-
"""
REAL SHAP feature-importance for the predictive-maintenance model, computed on
the daily maintenance data (Table3_Daily_Maintenance_Data.csv).

Design (leakage-free, matches Table 9 / Section 4.9):
  Target : Availability(t)  [%]   -- predicted ONE STEP AHEAD
  Inputs : yesterday's reliability state + today's calendar
           lag_failures = Failures(t-1)        (autoregressive history)
           operating    = Operating(t-1) [min] (exposure)
           MTBF         = MTBF(t-1) [min]
           MTTR         = MTTR(t-1) [min]
           downtime     = Downtime(t-1) [min]
           month        = calendar month (1-12)  (seasonality)
           year_index   = Year - min(Year)        (trend)
  No contemporaneous Operating(t)/Downtime(t) is used, so Availability is NOT
  algebraically leaked; the model genuinely forecasts it.

Outputs (written next to this script):
  shap_importance.csv   feature, mean_abs_shap, share_pct, rank
  shap_bar.png          global mean|SHAP| bar
  shap_beeswarm.png     per-instance summary
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "Table3_Daily_Maintenance_Data.csv")

daily = pd.read_csv(CSV)
daily.columns = [c.strip() for c in daily.columns]

MONTHS = {m: i + 1 for i, m in enumerate(
    ["January","February","March","April","May","June",
     "July","August","September","October","November","December"])}
daily["month_num"] = daily["Month"].map(MONTHS)

# --- aggregate daily -> MONTHLY (the resolution the paper's NARX is trained on;
#     the daily series was Poisson-disaggregated, so only the monthly signal is real)
df = (daily.groupby(["Year", "month_num"], as_index=False)
            .agg(Failures=("Failures", "sum"),
                 Downtime=("Downtime (min)", "sum"),
                 Operating=("Operating (min)", "sum")))
df = df.sort_values(["Year", "month_num"]).reset_index(drop=True)
df["MTBF"] = df["Operating"] / df["Failures"].replace(0, np.nan)
df["MTTR"] = df["Downtime"] / df["Failures"].replace(0, np.nan)
df["Availability (%)"] = (df["Operating"] - df["Downtime"]) / df["Operating"] * 100.0
df["MTTR"] = df["MTTR"].fillna(0.0)
df["MTBF"] = df["MTBF"].fillna(df["Operating"])
print(f"Monthly records: {len(df)}  ({df['Year'].min()}-{df['Year'].max()})")

# build one-step-ahead MONTHLY features (state at month t-1) + calendar at t
feat = pd.DataFrame()
feat["Lagged failure count (autoregressive history)"] = df["Failures"].shift(1)
feat["Operating hours (exposure)"]                    = df["Operating"].shift(1)
feat["MTBF"]                                          = df["MTBF"].shift(1)
feat["MTTR"]                                          = df["MTTR"].shift(1)
feat["Downtime"]                                      = df["Downtime"].shift(1)
feat["Calendar month (seasonality)"]                  = df["month_num"]
feat["Year / trend index"]                            = df["Year"] - df["Year"].min()

y = df["Failures"]   # NARX core task: forecast next-month failure count
ok = feat.notna().all(axis=1) & y.notna()
X, y = feat[ok].reset_index(drop=True), y[ok].reset_index(drop=True)
print(f"Samples: {len(X)}   Features: {list(X.columns)}")

from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                     subsample=0.9, colsample_bytree=1.0, random_state=0)
model.fit(X, y)                       # fit on full monthly series
from sklearn.metrics import r2_score, mean_absolute_error
pred = model.predict(X)
print(f"In-sample fit: R2={r2_score(y,pred):.3f}  MAE={mean_absolute_error(y,pred):.4f}")

import shap
explainer = shap.TreeExplainer(model)
sv = explainer(X)                     # global importance over the full series

mean_abs = np.abs(sv.values).mean(axis=0)
rank = (pd.DataFrame({"feature": list(X.columns), "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
rank["share_pct"] = 100 * rank["mean_abs_shap"] / rank["mean_abs_shap"].sum()
rank.index = np.arange(1, len(rank) + 1); rank.index.name = "rank"
print("\n=== GLOBAL FEATURE IMPORTANCE (mean|SHAP|) ===")
print(rank.round(4).to_string())
rank.round(6).to_csv(os.path.join(HERE, "shap_importance.csv"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
shap.plots.bar(sv, show=False, max_display=len(X.columns))
plt.tight_layout(); plt.savefig(os.path.join(HERE, "shap_bar.png"), dpi=200, bbox_inches="tight"); plt.close()
shap.summary_plot(sv, X, show=False)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "shap_beeswarm.png"), dpi=200, bbox_inches="tight"); plt.close()
print("\nSaved: shap_importance.csv, shap_bar.png, shap_beeswarm.png")
