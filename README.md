# Adaptive Physics-Informed Digital Twin for Predictive Maintenance — Reproducibility Package

This repository contains the code, data and computed results that reproduce the
quantitative experiments reported in the paper:

> **Prediction of Maintenance Failures and Decision-Making Based on Digital Twins:
> A Fleet-Wide Study Across Eleven Production-Line Machines.**

It accompanies the manuscript so that every numerical result — the model benchmark
(Table 5), the modern Transformer baselines, the SHAP explainability analysis
(Table 9, Figs. 15–18), and the statistical-significance / ablation / sensitivity
analyses (Tables 10–11) — can be regenerated from the raw maintenance records.

---

## Repository structure

```
DigitalTwin_PdM_Reproducibility/
├── code/
│   ├── benchmark_run.py          # Table 5: fair benchmark of the proposed adaptive
│   │                             #   physics-informed hybrid vs naive / statistical /
│   │                             #   ML / deep baselines, two evaluation regimes
│   ├── transformer_baselines.py  # Modern Transformer baselines (vanilla Transformer
│   │                             #   + PatchTST) added to Table 5
│   ├── shap_run.py               # Global SHAP importance (Table 9, Fig. 15 bar)
│   ├── shap_extra_plots.py       # SHAP summary (beeswarm, Fig. 16), dependence
│   │                             #   (Fig. 17) and force plot (Fig. 18)
│   └── ablation_sensitivity.py   # Tables 10–11: ablation of the adaptive gate +
│                                 #   sensitivity to gate hyper-parameters, with
│                                 #   bootstrap CIs, effect sizes and significance tests
├── data/
│   ├── Table3_Daily_Maintenance_Data.csv          # metal-detector machine (Machine_01)
│   ├── fleet_daily_maintenance_anonymized.csv     # all 11 machines, anonymized, tidy
│   └── DATA_DICTIONARY.md                          # columns, units, anonymization note
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
python code/benchmark_run.py          # -> Table 5 (baselines + proposed)
python code/transformer_baselines.py  # -> Transformer + PatchTST rows of Table 5
python code/shap_run.py               # -> Table 9 + Fig. 15 (global SHAP)
python code/shap_extra_plots.py       # -> Figs. 16–18 (summary / dependence / force)
python code/ablation_sensitivity.py   # -> Tables 10–11 + CIs, effect sizes, p-values
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

---

## Environment

Developed and tested with Python 3.11 on Windows 10. Key package versions are pinned
in `requirements.txt`. On Windows, set `n_jobs=1` for XGBoost (already done in the
scripts) to avoid a thread-deadlock observed with the default thread pool.

## License

Code is released under the MIT License (see `LICENSE`). If you use this code or data,
please cite the paper and this repository (see `CITATION.cff`).
