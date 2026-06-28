# -*- coding: utf-8 -*-
"""
Investment appraisal for the low-cost Digital-Twin predictive-maintenance system:
Payback Period, ROI, Net Present Value (NPV) and Benefit-Cost Ratio (BCR).

The ONLY measured quantity is the availability gain (Table 6): the metal-detector
machine recovers ~420 h of production per year. Everything monetary is an explicitly
INDICATIVE assumption (clearly stated), so the appraisal is a transparent scenario
analysis, not a claim of measured cash flows. All formulas are standard.

Base case (single machine = metal detector). Because ONE installation also serves the
other ten machines, fleet-level economics are strictly more favourable; the single-
machine case is therefore the conservative bound.
"""
import numpy as np

# ---------------- INDICATIVE ASSUMPTIONS (state clearly in the paper) -----------
CAPEX = 180_000      # EGP one-off: hardware 40k + software 20k + development 100k + training 20k
OPEX  = 18_000       # EGP/yr: software upkeep, recalibration, model maintenance (~10% of CapEx)
HOURS_RECOVERED = 420  # h/yr  (measured: from the 4.80 pp availability gain, Table 6)
N     = 5            # appraisal horizon (years)

def appraise(downtime_cost, r):
    benefit = HOURS_RECOVERED * downtime_cost           # gross annual saving
    ncf     = benefit - OPEX                             # net annual cash flow
    # annuity present-value factor
    paf = (1 - (1+r)**(-N)) / r
    pv_benefits = benefit * paf
    pv_costs    = CAPEX + OPEX * paf
    npv  = ncf * paf - CAPEX
    bcr  = pv_benefits / pv_costs
    payback = CAPEX / ncf                                # simple payback (years)
    roi_total = (ncf * N - CAPEX) / CAPEX * 100          # ROI over the horizon (%)
    roi_ann   = ncf / CAPEX * 100                        # simple annual ROI (%)
    return dict(benefit=benefit, ncf=ncf, payback=payback, npv=npv, bcr=bcr,
                roi_total=roi_total, roi_ann=roi_ann)

print(f"Assumptions: CapEx={CAPEX:,} EGP | OpEx={OPEX:,} EGP/yr | "
      f"recovered={HOURS_RECOVERED} h/yr | horizon={N} yr\n")

print("BASE CASE  (downtime cost = EGP 200/h, discount rate = 15%)")
b = appraise(200, 0.15)
print(f"  Gross annual saving : EGP {b['benefit']:,.0f}")
print(f"  Net annual cash flow: EGP {b['ncf']:,.0f}")
print(f"  Payback period      : {b['payback']:.2f} years")
print(f"  ROI (5-year)        : {b['roi_total']:.0f} %   (annual {b['roi_ann']:.0f} %)")
print(f"  NPV (5 yr, 15%)     : EGP {b['npv']:,.0f}")
print(f"  Benefit-Cost Ratio  : {b['bcr']:.2f}")

print("\nSENSITIVITY")
print(f"{'downtime EGP/h':>14} {'rate':>6} {'payback(yr)':>12} {'NPV(EGP)':>12} {'BCR':>6} {'ROI5(%)':>8}")
for dc in (200, 400):
    for r in (0.10, 0.15, 0.20):
        a = appraise(dc, r)
        print(f"{dc:>14} {r:>6.0%} {a['payback']:>12.2f} {a['npv']:>12,.0f} {a['bcr']:>6.2f} {a['roi_total']:>8.0f}")
