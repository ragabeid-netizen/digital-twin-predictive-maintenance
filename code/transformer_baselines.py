# -*- coding: utf-8 -*-
"""
Modern Transformer time-series baselines for the monthly availability forecast,
added per reviewer request (compare against recent Transformer-family models, not
only LSTM/GRU/ARIMA/XGBoost).

Two models are implemented from scratch in PyTorch and trained/evaluated with the
EXACT same monthly pipeline, windows, splits, seeds and metrics as benchmark_run.py
so the numbers drop straight into Table 5:

  1. Vanilla time-series Transformer  (multivariate encoder + attention pooling)
  2. PatchTST                          (patch embedding + channel-independent encoder)

Same two regimes:
  STATIONARY  : train 2010-2018, test 2019-2021
  REGIME-SHIFT: train 2010-2021, test 2022-2025
Each model averaged over 3 seeds (mean over seeds), early-stopping on a tail split.
NO fabricated numbers - whatever it prints is what it got.
"""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "Table3_Daily_Maintenance_Data.csv")

import torch, torch.nn as nn
torch.set_num_threads(1)

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
print(f"monthly rows: {len(df)}  years: {df.Year.min()}-{df.Year.max()}")

def metrics(yt,yp):
    yt,yp=np.asarray(yt,float),np.asarray(yp,float)
    mae=np.mean(np.abs(yt-yp)); rmse=np.sqrt(np.mean((yt-yp)**2))
    ss=np.sum((yt-yt.mean())**2); r2=1-np.sum((yt-yp)**2)/ss if ss>0 else float('nan')
    mape=np.mean(np.abs((yt-yp)/yt))*100
    return mae,rmse,r2,mape

def split(train_end_year, test_years):
    tr=df.index[df.Year<=train_end_year]
    te=df.index[df.Year.isin(test_years)]
    return list(tr),list(te)

def make_seq(idx, window=12):
    cols=["Avail","Failures","Downtime","Operating","MTTR"]
    data=df[cols].values
    Xs,ys,ti=[],[],[]
    for i in idx:
        if i-window<0: continue
        Xs.append(data[i-window:i]); ys.append(df.Avail.values[i]); ti.append(i)
    return np.array(Xs,dtype=np.float32),np.array(ys,dtype=np.float32),ti

# ---------- models ----------
class VanillaTST(nn.Module):
    """Standard Transformer encoder over the lookback window + attention pooling."""
    def __init__(self,nf,window,d_model=32,nhead=4,layers=2,ff=64,drop=0.1):
        super().__init__()
        self.inp=nn.Linear(nf,d_model)
        self.pos=nn.Parameter(torch.randn(1,window,d_model)*0.02)
        enc=nn.TransformerEncoderLayer(d_model,nhead,ff,drop,batch_first=True)
        self.enc=nn.TransformerEncoder(enc,layers)
        self.head=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,1))
    def forward(self,x):
        h=self.inp(x)+self.pos
        h=self.enc(h)
        h=h.mean(dim=1)                 # temporal pooling
        return self.head(h).squeeze(-1)

class PatchTST(nn.Module):
    """Channel-independent patch Transformer (Nie et al., 2023, ICLR)."""
    def __init__(self,nf,window,patch=4,stride=2,d_model=32,nhead=4,layers=2,ff=64,drop=0.1):
        super().__init__()
        self.nf=nf; self.patch=patch; self.stride=stride
        self.np_=(window-patch)//stride+1
        self.embed=nn.Linear(patch,d_model)
        self.pos=nn.Parameter(torch.randn(1,self.np_,d_model)*0.02)
        enc=nn.TransformerEncoderLayer(d_model,nhead,ff,drop,batch_first=True)
        self.enc=nn.TransformerEncoder(enc,layers)
        self.flatten=nn.Flatten(start_dim=1)
        self.head=nn.Linear(nf*self.np_*d_model,1)
    def forward(self,x):                 # x: (B,window,nf)
        B=x.shape[0]
        x=x.permute(0,2,1)              # (B,nf,window)  channel-independent
        # unfold into patches along time
        patches=x.unfold(dimension=2,size=self.patch,step=self.stride)  # (B,nf,np,patch)
        h=self.embed(patches)          # (B,nf,np,d_model)
        h=h.reshape(B*self.nf,self.np_,-1)+self.pos
        h=self.enc(h)                  # (B*nf,np,d_model)
        h=h.reshape(B,-1)
        return self.head(h).squeeze(-1)

def train_eval(MakeModel,name,tr,te,window=12,seeds=(0,1,2),epochs=300,lr=1e-3):
    Xtr,ytr,_=make_seq(tr,window); Xte,yte,tei=make_seq(te,window)
    if len(Xtr)<8 or len(Xte)==0:
        return None
    ns,nt,nf=Xtr.shape
    # standardize features and target on train stats
    mu=Xtr.reshape(-1,nf).mean(0); sd=Xtr.reshape(-1,nf).std(0)+1e-8
    Xtr2=(Xtr-mu)/sd; Xte2=(Xte-mu)/sd
    ymu,ysd=ytr.mean(),ytr.std()+1e-8
    ytr2=(ytr-ymu)/ysd
    # tail validation split for early stopping
    nval=max(2,int(0.15*len(Xtr2)))
    Xt,yt=Xtr2[:-nval],ytr2[:-nval]; Xv,yv=Xtr2[-nval:],ytr2[-nval:]
    preds=[]
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        m=MakeModel(nf,window)
        opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=1e-4)
        lossf=nn.SmoothL1Loss()
        Xtt=torch.tensor(Xt); ytt=torch.tensor(yt)
        Xvv=torch.tensor(Xv); yvv=torch.tensor(yv)
        Xee=torch.tensor(Xte2)
        best=1e9; best_state=None; bad=0
        for ep in range(epochs):
            m.train(); opt.zero_grad()
            out=m(Xtt); loss=lossf(out,ytt); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                vl=lossf(m(Xvv),yvv).item()
            if vl<best-1e-5: best=vl; best_state={k:v.clone() for k,v in m.state_dict().items()}; bad=0
            else:
                bad+=1
                if bad>=40: break
        if best_state: m.load_state_dict(best_state)
        m.eval()
        with torch.no_grad():
            p=m(Xee).numpy()*ysd+ymu
        preds.append(p)
    yp=np.mean(preds,axis=0)
    return metrics(yte,yp)

for label,(tey,tyrs) in {"STATIONARY (train<=2018, test 2019-2021)":(2018,[2019,2020,2021]),
                          "REGIME-SHIFT (train<=2021, test 2022-2025)":(2021,[2022,2023,2024,2025])}.items():
    tr,te=split(tey,tyrs)
    print(f"\n================ {label} ================")
    for name,Mk in [("Transformer",lambda nf,w:VanillaTST(nf,w)),
                    ("PatchTST",   lambda nf,w:PatchTST(nf,w))]:
        r=train_eval(Mk,name,tr,te)
        if r is None:
            print(f"  {name:14s} (insufficient data)")
        else:
            mae,rmse,r2,mape=r
            print(f"  {name:14s} MAE={mae:7.4f}  RMSE={rmse:7.4f}  R2={r2:8.3f}  MAPE={mape:7.4f}")
print("\nDONE.")
