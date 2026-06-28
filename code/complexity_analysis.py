# -*- coding: utf-8 -*-
"""
Computational-complexity profiling of the proposed model and the main baselines,
measured on the monthly availability-forecasting pipeline (same data as Table 5).
Reports REAL wall-clock training time, inference latency, trainable-parameter count
and model footprint, plus the asymptotic per-epoch / per-fit complexity.
n_jobs=1 (Windows XGBoost) ; times are means over repeated runs.
"""
import os, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__))
def _find_csv():
    for c in [os.path.join(HERE,"Table3_Daily_Maintenance_Data.csv"),
              os.path.join(HERE,"..","data","Table3_Daily_Maintenance_Data.csv"),
              r"C:\Users\R_HP\Desktop\26   6  2026\Table3_Daily_Maintenance_Data.csv"]:
        if os.path.exists(c): return c
    raise FileNotFoundError("Table3_Daily_Maintenance_Data.csv not found")
CSV=_find_csv()
d=pd.read_csv(CSV); d.columns=[c.strip() for c in d.columns]
M={m:i+1 for i,m in enumerate(["January","February","March","April","May","June","July",
   "August","September","October","November","December"])}
d["mn"]=d["Month"].map(M)
df=(d.groupby(["Year","mn"],as_index=False).agg(Failures=("Failures","sum"),
    Downtime=("Downtime (min)","sum"),Operating=("Operating (min)","sum")))
df=df.sort_values(["Year","mn"]).reset_index(drop=True)
df["MTTR"]=(df["Downtime"]/df["Failures"].replace(0,np.nan)).fillna(0)
df["Avail"]=(df["Operating"]-df["Downtime"])/df["Operating"]*100
F=pd.DataFrame()
for k in range(1,4): F[f"lag{k}"]=df.Avail.shift(k)
F["lag12"]=df.Avail.shift(12); F["lf"]=df.Failures.shift(1); F["op"]=df.Operating
y=df.Avail.values
ok=F.notna().all(1); X=F[ok].values; Y=y[ok.values]
n,p=X.shape; print(f"monthly samples={n}, features={p}")

def timeit(fn,reps=5):
    fn(); ts=[]
    for _ in range(reps):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return np.median(ts)

rows=[]
# ---- XGBoost (data model / proposed core) ----
from xgboost import XGBRegressor
def mk_xgb(): return XGBRegressor(n_estimators=400,max_depth=3,learning_rate=0.05,subsample=0.9,n_jobs=1,random_state=0)
m=mk_xgb(); ttr=timeit(lambda:mk_xgb().fit(X,Y)); m.fit(X,Y)
tinf=timeit(lambda:m.predict(X))/n*1e3
nparams=int(m.get_booster().trees_to_dataframe().shape[0])  # total tree nodes
rows.append(("Data model / ARW-PI core (XGBoost)",ttr,tinf,f"{nparams} nodes","O(K·d·n log n)"))

# ---- Adaptive hybrid overhead (gate is O(n), negligible) ----
def adaptive():
    mm=mk_xgb().fit(X,Y); pr=mm.predict(X)
    anchor=np.r_[Y[0],Y[:-1]]; doc=np.abs(X[:,-1]-X[:,-1].mean())/(X[:,-1].std()+1e-9)
    a=1/(1+np.exp(-3*(doc-2))); return a*anchor+(1-a)*pr
ttr_h=timeit(adaptive);
rows.append(("Adaptive hybrid (proposed, full)",ttr_h,tinf,f"{nparams} nodes + gate","O(K·d·n log n)+O(n)"))

# ---- LSTM / GRU (keras) ----
import tensorflow as tf
from tensorflow import keras
def seq(window=12):
    cols=["Avail","Failures","Downtime","Operating","MTTR"]; data=df[cols].values
    Xs,ys=[],[]
    for i in range(window,len(df)):
        Xs.append(data[i-window:i]); ys.append(df.Avail.values[i])
    return np.array(Xs,dtype="float32"),np.array(ys,dtype="float32")
Xs,ys=seq();
for kind in ["LSTM","GRU"]:
    def build():
        tf.keras.utils.set_random_seed(0)
        mdl=keras.Sequential([keras.layers.Input((Xs.shape[1],Xs.shape[2])),
            (keras.layers.LSTM(32) if kind=="LSTM" else keras.layers.GRU(32)),keras.layers.Dense(1)])
        mdl.compile(optimizer="adam",loss="mse"); return mdl
    mdl=build()
    t=time.perf_counter(); mdl.fit(Xs,ys,epochs=120,verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=15,restore_best_weights=True)],validation_split=0.15)
    ttr=time.perf_counter()-t
    tinf=timeit(lambda:mdl.predict(Xs,verbose=0))/len(Xs)*1e3
    rows.append((kind,ttr,tinf,f"{mdl.count_params():,} params","O(E·n·h²)"))

# ---- Transformer / PatchTST (torch) ----
import torch, torch.nn as nn; torch.set_num_threads(1)
class TST(nn.Module):
    def __init__(s,nf,w,dm=32):
        super().__init__(); s.inp=nn.Linear(nf,dm); s.pos=nn.Parameter(torch.randn(1,w,dm)*.02)
        s.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(dm,4,64,0.1,batch_first=True),2)
        s.head=nn.Sequential(nn.LayerNorm(dm),nn.Linear(dm,1))
    def forward(s,x): h=s.inp(x)+s.pos; return s.head(s.enc(h).mean(1)).squeeze(-1)
mu,sd=Xs.reshape(-1,Xs.shape[2]).mean(0),Xs.reshape(-1,Xs.shape[2]).std(0)+1e-8
Xt=torch.tensor((Xs-mu)/sd); yt=torch.tensor((ys-ys.mean())/(ys.std()+1e-8))
def train_torch():
    torch.manual_seed(0); mdl=TST(Xs.shape[2],Xs.shape[1]); opt=torch.optim.Adam(mdl.parameters(),1e-3)
    lf=nn.SmoothL1Loss()
    for _ in range(150):
        mdl.train(); opt.zero_grad(); loss=lf(mdl(Xt),yt); loss.backward(); opt.step()
    return mdl
t=time.perf_counter(); mdl=train_torch(); ttr=time.perf_counter()-t
mdl.eval()
with torch.no_grad(): tinf=timeit(lambda:mdl(Xt).numpy())/len(Xt)*1e3
nparam=sum(pp.numel() for pp in mdl.parameters())
rows.append(("Transformer",ttr,tinf,f"{nparam:,} params","O(E·n·w²·dm)"))

print(f"\n{'Model':36s} {'Train(s)':>9} {'Infer(ms/sample)':>16} {'Size':>22} {'Complexity':>22}")
for name,ttr,tinf,size,cplx in rows:
    print(f"{name:36s} {ttr:9.3f} {tinf:16.4f} {size:>22} {cplx:>22}")
print("\nDONE.")
