# Adaptive Reliability-Weighted Digital Twin for Predictive Maintenance — Reproducibility Package

This repository contains the code, data and computed results that reproduce the
quantitative experiments reported in the paper:

> **An Adaptive Reliability-Weighted Digital Twin for Predictive Maintenance under
> Operating-Regime Shift: A Sixteen-Year Industrial Case Study.**

It accompanies the manuscript so that every numerical result — the model benchmark
(Table 2), the modern Transformer baselines, the SHAP explainability analysis
(Fig. 6 and Supplementary Table S4), the ablation study (Table 3), the external
validation on the public AI4I 2020 benchmark (Table 4), and the sensitivity grid
(Supplementary Table S5) — can be regenerated from the raw maintenance records.

> **Note on an earlier version of this README.** A previous revision described the
> work as *"A Fleet-Wide Study Across Eleven Production-Line Machines."* That framing
> was withdrawn during manuscript revision and is incorrect. See **Scope** below.

---

## Scope — what is measured and what is not

The measured evidence is **one machine**: the metal detector on a biscuit production
line at a food-manufacturing plant in 6th of October City, Egypt. Its maintenance
logbook covers **sixteen complete years, January 2010 – December 2025 (192 monthly
records)**, plus a partial 2026 annual aggregate that is reported for completeness
and **excluded from every headline claim**.

The repository also contains **ten synthetic replicas**. These are **proportional
scalings of that single measured series** by fixed per-machine factors (0.80–0.98),
with the operating calendar held constant. They exist to show that the data pipeline
runs unchanged at fleet scale. **They are not independent assets and carry no
independent evidential weight.** Any aggregate computed across the eleven assets
reflects the scaling factors, not eleven independent failure processes.

---

## What the model actually is

`ARW-PI` blends two predictors **at inference time**:

```
y_hybrid(t) = α(t)·y_phys(t) + [1 − α(t)]·y_ann(t)
α(t)        = σ( s·[ δ_oc(t) − τ ] )
```

* `y_ann` — the data-driven component: a **gradient-boosted regressor** fitted under
  the **standard squared-error objective**.
* `y_phys` — the reliability anchor, from the renewal identity
  `A = MTBF / (MTBF + MTTR)`. Evaluated at the last observed state this returns the
  last observed availability, so **the anchor coincides numerically with a
  persistence forecast**. This is a property of the identity, and the paper says so.
* `δ_oc(t)` — an operating-condition novelty index. The gate shifts weight to the
  anchor as the regime moves away from the training distribution.

> **Important.** Section 3.5.2 of the paper specifies a physics-informed *training*
> objective (Eqs. 7–10). **That extension is specified but not evaluated, and it is
> not implemented in this repository.** The code implements the inference-time form
> above only. No result in the paper depends on Eqs. (7)–(10).

---

## Repository structure

```
DigitalTwin_PdM_Reproducibility/
├── code/
│   ├── benchmark_run.py          # Table 2: benchmark of the proposed adaptive
│   │                             #   reliability-weighted hybrid vs naive /
│   │                             #   statistical / ML / deep baselines,
│   │                             #   two evaluation regimes
│   ├── transformer_baselines.py  # Transformer + PatchTST rows of Table 2
│   ├── shap_run.py               # Global SHAP importance (Fig. 6,
│   │                             #   Supplementary Table S4)
│   ├── shap_extra_plots.py       # SHAP summary (beeswarm), dependence and force
│   │                             #   plots — Supplementary Figs. S1–S3
│   ├── ablation_sensitivity.py   # Table 3 (ablation) and Supplementary Table S5
│   │                             #   (gate sensitivity), with bootstrap CIs,
│   │                             #   effect sizes and significance tests
│   ├── economic_analysis.py      # Section 4.12 investment appraisal — payback,
│   │                             #   ROI, NPV, benefit–cost ratio and the
│   │                             #   sensitivity grid (Supplementary Tables S6, S8)
│   ├── external_validation_ai4i.py # Table 4: external validation of the adaptive
│   │                             #   gate on the public AI4I 2020 benchmark
│   ├── complexity_analysis.py    # Supplementary Table S7: training/inference time,
│   │                             #   parameters, memory and asymptotic complexity
│   └── graphical_abstract.py     # the graphical-abstract / TOC figure
├── data/
│   ├── Table3_Daily_Maintenance_Data.csv       # the ONE measured machine
│   │                                           #   (metal detector, Machine_01)
│   ├── fleet_daily_maintenance_anonymized.csv  # Machine_01 (measured) plus ten
│   │                                           #   SYNTHETIC proportional replicas
│   │                                           #   — see Scope above
│   └── DATA_DICTIONARY.md                      # columns, units, anonymization note
├── results/
│   ├── benchmark_results.txt     # saved console output of benchmark_run.py
│   ├── transformer_results.txt   # saved console output of transformer_baselines.py
│   └── shap_importance.csv       # computed mean|SHAP| ranking
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## Quick start

```bash
# 1. Create and activate a clean environment (Python 3.11 recommended)
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Reproduce the results
python code/benchmark_run.py          # -> Table 2 (baselines + proposed)
python code/transformer_baselines.py  # -> Transformer + PatchTST rows of Table 2
python code/shap_run.py               # -> Fig. 6 + Supplementary Table S4
python code/shap_extra_plots.py       # -> Supplementary Figs. S1–S3
python code/ablation_sensitivity.py   # -> Table 3 + Supplementary Table S5
python code/external_validation_ai4i.py # -> Table 4
```

All scripts read the dataset from `data/Table3_Daily_Maintenance_Data.csv`
(by default they expect the CSV next to the script — adjust the `CSV` path
constant at the top of each file, or place a copy in `code/`).

---

## DATA NOTE (important for correct interpretation)

All models are **trained and evaluated on the measured MONTHLY maintenance records**
(192 machine-months, 2010–2025). The daily-resolution CSV is a Poisson-disaggregation
of those monthly totals and is used **only** to drive the Digital-Twin real-time
simulation layer — it is **never** used for model training or evaluation. Every script
therefore aggregates the daily file to monthly before fitting, exactly as described in
the paper. This keeps the reported accuracy free of any synthetic-data leakage.

R² is negative for all models in the 2022–2025 regime-shift window because the
availability series there is nearly saturated (very low variance); MAE and RMSE are
the primary metrics, as stated in the paper.

Under the 2022–2025 shift the proposed estimator's error is **statistically
indistinguishable from the persistence anchor it falls back to** (p = 0.13), and
significantly lower than every learned forecaster (p < 0.01). That is the intended
behaviour of the gate, and the paper reports it as such.

---

## Environment

Developed and tested with Python 3.11 on Windows 10. Key package versions are pinned
in `requirements.txt`. On Windows, set `n_jobs=1` for XGBoost (already done in the
scripts) to avoid a thread-deadlock observed with the default thread pool.

## License

Code is released under the MIT License (see `LICENSE`). If you use this code or data,
please cite the paper and this repository (see `CITATION.cff`).
