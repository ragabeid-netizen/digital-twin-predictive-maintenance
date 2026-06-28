# Data dictionary and availability note

This folder holds the maintenance dataset used in the paper, provided in **anonymized**
form for open Data Availability.

## Anonymization statement

The data contain **no company name, no location, no operator or personal data, and no
commercial identifiers** — only calendar dates and aggregate machine-reliability counts.
Machines are labelled generically as `Machine_01 … Machine_11` with a generic machine
*type* only (Metal detector / Oven / Packaging / Kneading). No proprietary brand or model
name is included. The dataset can therefore be shared and reused without restriction.

## Files

| File | Description |
|------|-------------|
| `Table3_Daily_Maintenance_Data.csv` | Daily reliability indicators for the metal-detector machine (`Machine_01`), 2010–2025. This is the machine used for the detailed model analysis (benchmark, SHAP). |
| `fleet_daily_maintenance_anonymized.csv` | The full anonymized fleet: all 11 machines stacked in tidy (long) format, 2010–2025. |

## Columns

| Column | Unit | Meaning |
|--------|------|---------|
| `Machine_ID` | — | Anonymized machine identifier (`Machine_01`…`Machine_11`). |
| `Machine_type` | — | Generic type: Metal detector, Oven, Packaging, Kneading. |
| `Year` | year | Calendar year (2010–2025). |
| `Month` | — | Calendar month name. |
| `Day` | day | Day of month. |
| `Failures` | count | Number of failures on that day. |
| `Downtime_min` | minutes | Total downtime on that day. |
| `Operating_min` | minutes | Planned/operating time on that day. |
| `MTBF_min` | minutes | Mean time between failures (Operating ÷ Failures). |
| `MTTR_min` | minutes | Mean time to repair (Downtime ÷ Failures). |
| `Availability_pct` | % | (Operating − Downtime) ÷ Operating × 100. |

(`Table3_Daily_Maintenance_Data.csv` uses the original header style
`Downtime (min)`, `Operating (min)`, etc.; the fleet file uses the cleaned names above.)

## Provenance and correct use (important)

- The **measured signal is monthly.** All model training and evaluation in the paper use
  the **monthly aggregation** of these records — never the daily rows.
- `Machine_01` (metal detector) is the **real measured baseline.** The other ten machines'
  daily series are derived from that baseline by fixed percentage scaling with **Poisson
  disaggregation**, calibrated so that the **monthly totals are exact**. The daily
  resolution drives only the Digital-Twin real-time simulation; it is not used to fit or
  evaluate any model.
- Aggregating the fleet file to monthly reproduces the fleet totals reported in the paper
  (e.g. 64,603 failures across the eleven machines over 2010–2025).
