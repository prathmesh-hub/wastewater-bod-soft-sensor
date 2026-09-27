# Wastewater BOD Soft Sensor

A neural network-based **soft sensor** that estimates effluent **Biochemical Oxygen Demand (BOD5)** in a wastewater treatment plant using fast, real-time process variables — eliminating the need to wait 5 days for a lab result.

## The Problem

BOD5 is the standard measure of organic pollution in treated wastewater, but testing it requires a **5-day biological incubation** — it cannot be measured instantly or automated. By the time a lab result comes back, the water has already been discharged, making real-time monitoring or corrective action impossible with the raw lab test alone.

This project builds a machine learning model that estimates BOD5 **immediately**, using other process variables that *can* be measured within minutes to a couple of hours (flow rate, Chemical Oxygen Demand, suspended solids, sediments, and conductivity).

## Dataset

- **Source:** [UCI Machine Learning Repository — Water Treatment Plant](https://archive.ics.uci.edu/dataset/106/water+treatment+plant) (DOI: `10.24432/C5FS4C`, CC BY 4.0)
- 527 daily records from a real urban wastewater treatment plant, 38 attributes
- Cleaned to 439 usable records after removing documented plant-fault days and rows with missing values

## Approach

| Step | Details |
|---|---|
| **Target** | `DBO-S` — effluent BOD5 (mg/L O₂) |
| **Inputs** | 13 fast-measurable process variables across influent, primary settler, and secondary settler stages (COD, solids, sediments, conductivity, flow) |
| **Preprocessing** | StandardScaler normalization, fault-day exclusion, complete-case removal for missing values |
| **Models compared** | Artificial Neural Network (13→16→8→1, ReLU, dropout), Linear Regression, Random Forest |
| **Validation** | 5-fold cross-validation (not a single train/test split) to ensure results are consistent, not a lucky split |

## Results

| Feature configuration | ANN R² | Linear Regression R² | Random Forest R² |
|---|---|---|---|
| Full 18 features | 0.007 | 0.123 | 0.076 |
| **Trimmed 13 features (final)** | 0.036 | **0.129** | 0.070 |
| Today + yesterday's readings (lag-1) | -0.035 | 0.108 | 0.103 |

**Key finding:** across every configuration tested — including a dedicated test of the plant's biological retention-time effect via lag features — Linear Regression matched or outperformed the more complex models. This indicates the accuracy ceiling comes from the information content of same-day/one-day process readings, not from model choice. See the full project report for the detailed methodology, root-cause analysis, and discussion.

## Repository Contents

```
├── water_treatment_final.csv          # Cleaned, modeling-ready dataset
├── step1_scale_and_split.py           # Feature scaling + train/test split
├── training_mod.py                    # ANN + Linear Regression training and evaluation
├── kfold_validation.py                # 5-fold cross-validation (full feature set)
├── generate_predictions.py            # Generates actual vs. predicted BOD table
├── soft_sensor_demo.html              # Interactive web demo (ANN + Linear Regression)
└── Soft_Sensor_Project_Report.pdf     # Full project report with methodology, results, and Q&A prep
```

## Running It Locally

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib joblib
python step1_scale_and_split.py
python training_mod.py
```

To view the demo, simply open `soft_sensor_demo.html` in any browser — it's fully self-contained, no server required.

## Limitations

- Model accuracy is modest (R² ≈ 0.13 at best) — this is a proof-of-concept demonstrating the soft-sensor approach, not a production-calibrated instrument.
- Dataset size (439 rows) is modest by machine learning standards.
- Scoped to normal plant operating conditions (documented fault days excluded).

## Future Work

- Test on larger, richer datasets (e.g. plants with additional biological indicators like ammonia and nitrogen)
- Explore longer historical lag windows (2–3+ days)
- Deploy with real physical sensors for live, continuous operation

## License

Dataset: CC BY 4.0 (UCI Machine Learning Repository). Code: for academic/educational use.
