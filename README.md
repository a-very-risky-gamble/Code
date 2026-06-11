# Propellant Service Life Prediction — Accelerated Aging ML Pipeline

## What this project does

Predicts the storage service life of a solid rocket propellant from accelerated aging test data. Small propellant samples are aged at elevated temperatures (50, 60, 70 °C) for varying numbers of days, then pulled to failure in a UTM machine. The code uses the Arrhenius equation to map those short high-temperature aging periods to equivalent years at the actual storage temperature (27 °C), trains a machine learning model on the resulting data, and finds when the propellant's strength drops below an acceptable threshold.

The predicted output is a single number: **service life in years at 27 °C**, with a GPR uncertainty band.

---

## File structure

```
project/
│
├── propellant_service_life_aging_ML.m   ← main script (run this)
├── aging_service_life_results.csv       ← output: averaged dataset + d_equiv
│
└── DATA/                                ← folder containing all ~500 xlsx files
    ├── T50_d0_v5_s1.xlsx
    ├── T50_d0_v5_s2.xlsx
    ├── T50_d7_v5_s1.xlsx
    ├── T60_d30_v50_s3.xlsx
    ├── T70_d90_v500_s2.xlsx
    └── ...
```

All data files must be in a single flat folder. Sub-folders are not scanned.

---

## File naming convention

Every xlsx file must be named exactly:

```
T{temp}_d{days}_v{speed}_s{sample}.xlsx
```

| Token | Meaning | Example values |
|-------|---------|----------------|
| T | Aging temperature [°C] | 50, 60, 70 |
| d | Days aged at that temperature | 0, 7, 14, 30, 60, 90, 120 … |
| v | Crosshead speed during the pull test [mm/min] | 5, 50, 500 |
| s | Sample / replicate number | 1, 2, 3 … |

Examples: `T50_d30_v5_s1.xlsx`, `T70_d90_v500_s3.xlsx`

Files that do not match this pattern are skipped with a warning.

---

## Data file format

Each xlsx file must have **three columns** in this order:

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| `disp_mm` | `load_N` | `t_min` |
| Displacement [mm] | Load [N] | Time [min] |

- One header row is expected. Extra blank rows are dropped automatically.
- Column names must match exactly (case-sensitive).

---

## Requirements

- MATLAB R2021a or later
- **Statistics and Machine Learning Toolbox** (required for `fitrgp`, `fitrensemble`, `fitrsvm`, `fitlm`, `crossvalind`)

No additional toolboxes or external packages are needed.

---

## Quick start

1. Copy all your xlsx files into one folder (e.g. `C:\Users\USER\DATA`).
2. Open `propellant_service_life_aging_ML.m`.
3. Edit the two lines at the top of Section 1:

```matlab
data_dir   = 'C:\Users\USER\DATA';         % ← your folder here
output_csv = 'aging_service_life_results.csv';
```

4. Optionally adjust the key parameters (see table below).
5. Run the script. All plots and the summary print automatically.

---

## Key parameters

These are all in **Section 1 — USER INPUTS** near the top of the script.

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `L0` | 47.75 mm | Gauge length of the test specimen |
| `A0` | 25 mm² | Cross-sectional area of the test specimen |
| `T_service_C` | 27 °C | Storage / operational temperature to predict life at |
| `threshold_frac` | 0.8 | Failure criterion — life ends when strength drops to this fraction of initial. 0.8 = 20 % strength loss |
| `v_ref_mmmin` | 5 mm/min | Strain rate used for the service life curve. Use the slowest rate (5) for a conservative prediction |

**If you change `A0`:** all stresses scale linearly with 1/A0. Use the actual measured cross-section area from your specimen drawing.

**If you change `threshold_frac`:** the sensitivity table in the output shows life at every threshold from 50 % to 95 %, so you can compare without re-running.

---

## What the script outputs

### Printed to the command window

```
Found 500 files.
Parsed 500 valid filenames.
Aging temperatures : [50 60 70] degC
Aging durations    : [0 7 14 30 60 90 120] days
...
Ea (chemical aging) = 97.43 kJ/mol

----- 5-Fold Cross-Validation (n=135) -----
Model                              R^2        RMSE(ksc)
----------------------------------------------------------
Linear                             0.8821     0.04231
Polynomial (deg 2)                 0.9104     0.03812
GPR Matern 5/2                     0.9463     0.02914   ← best
...
Best model: gpr_mat  (RMSE = 0.02914 ksc)

============================================================
PREDICTED SERVICE LIFE: 18.43 years at 27 degC
(equivalent aging time to 80% strength retention)
GPR uncertainty band  : 16.21 -- 21.07 years (±1 sigma)
============================================================
```

### Plots (4 figures)

| Figure | What it shows |
|--------|--------------|
| 1. Raw degradation | sigma_m vs aging days, one line per (temperature, strain rate). Shows the raw data before any Arrhenius shift. |
| 2. Master degradation curve | Same data after Arrhenius shift to equivalent days at 27 °C. All three temperature curves should collapse onto one. The ML fit is overlaid. If the curves do not collapse well, Ea may need reviewing. |
| 3. Predicted vs Actual | Each condition's measured sigma_m against the model's prediction. Points should lie close to the 45° line. R² and RMSE are the headline accuracy numbers. |
| 4. Service life curve | Predicted strength vs equivalent aging time at 27 °C with GPR uncertainty shading. The threshold line and the predicted service life marker are shown. |

### CSV file

`aging_service_life_results.csv` — one row per unique (T, d, v) condition after averaging replicates. Columns:

| Column | Units | Description |
|--------|-------|-------------|
| T_aging_degC | °C | Aging temperature |
| d_days | days | Aging duration in lab |
| v_mmmin | mm/min | Crosshead speed |
| d_equiv_days | days | Equivalent aging at 27 °C (via Arrhenius) |
| sigma_m_ksc | ksc | Mean peak stress |
| eps_m | — | Mean strain at peak |

---

## How the prediction works (brief)

**Step 1 — Extract strength.** From each file, stress σ = (load / A0) × 10.197 [ksc] and strain ε = disp / L0 are computed. The peak stress σ_m is the single number kept per test.

**Step 2 — Fit activation energy.** An Arrhenius activation energy Ea is found by searching for the value that best collapses the σ_m vs aging-time curves at all three temperatures onto one curve. This is done by minimising the residual scatter of a polynomial fit in log(d_equiv + 1) space. Ea is bounded to [40, 200] kJ/mol so it cannot be negative (degradation must be faster at higher temperature).

**Step 3 — Map to service temperature.**
```
d_equiv = d_lab × exp( Ea/R × (1/T_service_K − 1/T_aging_K) )
```
One day at 70 °C is typically worth 100–200 equivalent days at 27 °C depending on Ea.

**Step 4 — Train ML.** Six models are trained on features [log(d_equiv + 1), log(strain_rate)] to predict log(σ_m). The best model is selected by 5-fold cross-validation (lowest RMSE). A GPR model is always co-trained to provide uncertainty bounds.

**Step 5 — Predict life.** The best model sweeps equivalent aging time from 0 to 100 years at T_service. Service life is where the predicted strength crosses the threshold (default 80 % of initial).

---

## Sensitivity outputs

The script automatically prints three sensitivity tables after the main result:

1. **Threshold sensitivity** — life at every threshold from 50 % to 95 % retention.
2. **Strain rate sensitivity** — life at v = 5, 50, and 500 mm/min.
3. **Ea sensitivity** — life if Ea were 60, 70, 80 … 140 kJ/mol, regardless of the fitted value. This shows how much the fitted Ea drives the final answer.

These are printed for transparency — a large change in life across the Ea range means the dataset's temperature span needs to be wider to constrain Ea better.

---

## Known limitations

**Mechanical vs chemical aging.** This code models strength loss from chemical aging (binder oxidation, plasticiser migration, interface degradation). It does not model mechanical creep under sustained load. For typical HTPB-AP propellants stored at ambient conditions, chemical aging is the dominant life-limiting mechanism, so this is the appropriate model.

**Extrapolation beyond data.** The Arrhenius mapping assumes a single activation energy applies uniformly over the temperature range 27–70 °C. If different degradation mechanisms dominate at different temperatures, the mapping may not hold. Check Plot 2: if the three curves collapse cleanly, the assumption is valid for your data.

**Threshold definition.** The default threshold of 80 % (20 % strength loss) is a common industry criterion but is not universal. Check the specification for your propellant class and adjust `threshold_frac` accordingly.

**Replicates.** Samples at the same (T, d, v) condition are averaged before ML training. If replicates show very high scatter (CV > 15 %), investigate whether test conditions were consistent.

---

## Units

All stresses are in **ksc (kilogram-force per square centimetre)**.

```
1 ksc  =  0.0981 MPa  =  98.1 kPa  =  1 kgf/cm²
1 MPa  =  10.197 ksc
```

The conversion is applied once during data loading (`sigma_ksc = (load_N / A0) × 10.197`) and all subsequent calculations, printed values, and saved CSV values are in ksc.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| "No files matching pattern found" | Wrong `data_dir` or filenames don't match pattern | Check path and that files are named T{n}_d{n}_v{n}_s{n}.xlsx |
| Ea comes out < 50 kJ/mol | Very little temperature dependence in the data, or d=0 (unaged) files are missing | Check that aging temperatures span at least 20 °C; add d=0 reference tests if absent |
| Plot 2 curves do not collapse | Ea is uncertain or data at one temperature has outliers | Review raw data at each temperature; check for file read errors |
| Service life prints "> 100 years" | Degradation is very slow at chosen threshold, or threshold_frac is too low | Increase threshold_frac or extend the sweep range in Section 10 |
| GPR is slow | Large number of averaged conditions (n > 300) | GPR scales as O(n³); reduce by averaging more aggressively or switching to a faster model manually |
| `crossvalind` not found | Statistics and Machine Learning Toolbox not installed | Install the toolbox or replace `crossvalind` with a manual index partition |
