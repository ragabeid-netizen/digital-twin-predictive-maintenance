# -*- coding: utf-8 -*-
"""
Graphical Abstract / TOC figure for the paper. Single-panel horizontal pipeline
summarising data -> Digital Twin -> adaptive physics-informed model + SHAP -> outcomes,
with the real headline numbers. Elsevier size: ~1328 x 620 px (h x w proportion).
Outputs: graphical_abstract.png (and .pdf) next to this script's project root.
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE=os.path.dirname(os.path.abspath(__file__))
OUTdir=os.path.dirname(HERE)  # repo root
NAVY="#1F3864"; BLUE="#2E5C9A"; TEAL="#2C8C99"; GREEN="#2E8B57"
AMBER="#C77A0A"; LIGHT="#EEF2F8"; INK="#1A1A1A"

fig,ax=plt.subplots(figsize=(13.28,6.0),dpi=100)
ax.set_xlim(0,1328); ax.set_ylim(0,600); ax.axis("off")

# title
ax.text(664,572,"Adaptive Physics-Informed Digital Twin for Predictive Maintenance",
        ha="center",va="center",fontsize=16,fontweight="bold",color=NAVY)
ax.text(664,545,"of a biscuit production line — 11 machines, 16 years of records (2010–2025)",
        ha="center",va="center",fontsize=11.5,color=INK)

def box(x,y,w,h,title,lines,fc,tc="white",ts=12):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=6,rounding_size=14",
                 linewidth=0,facecolor=fc))
    ax.text(x+w/2,y+h-26,title,ha="center",va="center",fontsize=ts,fontweight="bold",color=tc)
    ax.text(x+w/2,y+(h-44)/2,"\n".join(lines),ha="center",va="center",fontsize=9.6,color=tc,linespacing=1.45)

def arrow(x0,x1,y):
    ax.add_patch(FancyArrowPatch((x0,y),(x1,y),arrowstyle="-|>",mutation_scale=22,
                 linewidth=2.4,color=NAVY))

yb=300; h=150
box(20 ,yb,250,h,"Production line",
    ["11 machines","Maintenance logs","Failures · downtime","MTBF · MTTR · availability"],NAVY)
arrow(276,300,yb+h/2)
box(304,yb,250,h,"Digital Twin",
    ["Factory I/O · Arduino","sensors · Modbus TCP","Power BI dashboard","(two-way coupling)"],BLUE)
arrow(560,584,yb+h/2)
box(588,yb,260,h,"Adaptive model",
    ["Physics-informed hybrid","ARW-PI adaptive gate","reliability + health index","trained w/ physics loss"],TEAL)
arrow(854,878,yb+h/2)
box(882,yb,250,h,"Explainability",
    ["SHAP attribution","MTBF, lagged failures,","MTTR, seasonality","summary·dependence·force"],GREEN)

# downward arrow from model cluster to outcomes
ax.add_patch(FancyArrowPatch((664,yb-4),(664,yb-44),arrowstyle="-|>",mutation_scale=22,
             linewidth=2.4,color=AMBER))

# outcomes banner
ax.add_patch(FancyBboxPatch((1158,yb),150,h,boxstyle="round,pad=6,rounding_size=14",
             linewidth=0,facecolor=AMBER))
ax.text(1233,yb+h-26,"Decisions",ha="center",va="center",fontsize=12,fontweight="bold",color="white")
ax.text(1233,yb+(h-44)/2,"predictive\nmaintenance\n&\nscheduling",ha="center",va="center",
        fontsize=9.6,color="white",linespacing=1.5)
arrow(1138,1158,yb+h/2)

# outcomes strip (bottom)
oy=70; ow=410; oh=150
def outcome(x,big,small,color):
    ax.add_patch(FancyBboxPatch((x,oy),ow,oh,boxstyle="round,pad=6,rounding_size=12",
                 linewidth=1.4,edgecolor=color,facecolor=LIGHT))
    ax.text(x+ow/2,oy+oh-34,big,ha="center",va="center",fontsize=15,fontweight="bold",color=color)
    ax.text(x+ow/2,oy+44,small,ha="center",va="center",fontsize=10.2,color=INK,linespacing=1.4)

outcome(28 ,"Availability  93.6% → 98.4%","metal-detector gain of +4.8 pp;\nsame DT pipeline applied to all 11 machines",NAVY)
outcome(459,"Forecast MAE ≈ 0.60%","robust under operating-regime shift\n(best-of-both vs Transformer, XGBoost, LSTM)",TEAL)
outcome(890,"Payback 2.7 yr · ROI 83%","positive NPV & benefit–cost ratio 1.17\n(conservative single-machine case)",AMBER)

plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
for ext in ("png","pdf"):
    plt.savefig(os.path.join(OUTdir,"graphical_abstract."+ext),dpi=150 if ext=="png" else None,
                bbox_inches="tight",facecolor="white")
print("saved graphical_abstract.png/.pdf in", OUTdir)
