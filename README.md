# VIGIL — learning what to retrieve from budget-constrained agricultural sensor networks

Code, configuration and result files for the paper

> **Value-of-Information Polling with Probabilistic Neural Graph Filters for 
>Budget-Constrained Soil Moisture Monitoring** .

VIGIL (*value-of-information-guided inference loop*) decides, every hour, which loggers of a soil-sensor
network a field gateway should poll when only `K` of `N` loggers can be read. A neural predict–update graph
filter estimates the full soil-moisture field (current value, look-ahead level and their variances) from the
polled readings; a Whittle-type value-of-information index turns the *reducible* look-ahead uncertainty and the
expected number of undetected wetting increments into polling decisions. The urgent (alert) and strategic
(maps, outlooks, model refresh) analytics are then dispatched across edge and co-tenant cloud resources whose
capacity follows real VM traces.

Everything is evaluated in **closed loop by exact trace replay** of the 42-logger, 5-depth CAF sensor archive:
a polled logger returns its archived reading, unpolled readings stay hidden and serve as ground truth.

## Repository layout

```
configs/            data.yaml (paths, QC, splits), train.yaml (models), experiments.yaml (evaluation grids)
src/vigil/
  data/             CAF loader + QC, weather (NASA POWER, SCAN), Bitbrains, dataset build
  sim/              retrieval memory (ages, availability, floors), edge-cloud discrete-event simulator
  models/           VIGIL network, neural baselines (MLP, GRU, Transformer, DCRNN, Graph WaveNet),
                    LightGBM, rain-conditioned Kalman filter, capacity forecaster
  engine.py         closed-loop rollout (training and evaluation), polling policies dispatch
  policies.py       VoI index (growth floors, shrinkage, urgent mass, Whittle index), RR, random, ARS, oracle
  train.py          training CLI (all neural models and LightGBM)
  evaluate.py       closed-loop evaluation and scoring
  metrics.py        RMSE/MAE/CRPS/coverage, event detection metrics
scripts/            experiment drivers, statistics, tables and figures (see below)
results/
  models/           trained models (model.pt / model.txt) and training logs per seed
  raw/              per-run evaluation outputs (JSON metrics + NPZ daily errors, onset delays, alerts)
  processed/        aggregated CSV files: <group>.csv (seed means), <group>_std.csv, <group>_seeds.csv
data/README.md      how to obtain the four public datasets
```

## Installation

```bash
conda env create -f environment.yml && conda activate vigil
# or: python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pip install -e .
```

All experiments run on CPU (one thread per process). The paper's runs used a 2-vCPU, 8-GB cloud VM.
Do not run more than two training or evaluation processes at once on 8 GB of RAM.

## Data

See [`data/README.md`](data/README.md) for download links and the expected layout under `data/raw/`
(or set `VIGIL_DATA=/path/to/raw`). Then build the processed tensors:

```bash
python -m vigil.data.build            # -> data/processed/caf_processed.npz, build_report.json, ...
python scripts/data_facts.py          # dataset table + numbers quoted in Section 3
python scripts/sample_rows.py         # verbatim example records (Table 2)
python scripts/compare_power_scan.py  # NASA POWER vs on-farm SCAN weather
```

## Reproducing the paper

```bash
# 1. train 50 models: 5 seeds of VIGIL and of each neural/GBDT baseline, 3 seeds of each VIGIL ablation.
#    Two restartable queues (finished jobs are skipped); about 6 h on a 2-vCPU machine.
bash scripts/train_queue.sh results/queues/jobs_1.txt &
bash scripts/train_queue.sh results/queues/jobs_2.txt &
#    single job:  python -m vigil.train --model vigil --variant default --seed 0

# 2. select the shrinkage weight kappa of the polling index on the validation year (2014)
python scripts/tune_policy.py

# 3. closed-loop evaluation on the test period (2015-01-01 .. 2016-06-16); about 5 h in total
for g in estimators kf_pipeline kf_voi policies baseline_ars tradeoff policy_ablations ablations weather; do
  python scripts/run_experiments.py --group $g
done

# 4. costs and scalability (run on an idle machine), capacity forecasting, edge-cloud case study, example trace
python scripts/measure_costs.py
python scripts/run_capacity.py
python scripts/run_edgecloud.py
python scripts/example_trace.py

# 5. aggregate, test, tabulate, plot
python scripts/aggregate.py
python scripts/stats_tests.py
python scripts/make_tables.py        # manuscript/tables/*.tex and manuscript/macros/results.tex
python scripts/make_figures.py       # manuscript/figures/*.pdf and manuscript/macros/figures.tex
python scripts/check_consistency.py  # macros, inputs, figures, citations and quoted settings

```


With 8 GB of RAM, run at most two training or evaluation processes at the same time (a VIGIL training
process needs about 2.5 GB).

| Paper element | Produced by |
|---|---|
| Table 1 (datasets), numbers in Sec. 3 | `scripts/data_facts.py` |
| Table 2 (sample records) | `scripts/sample_rows.py` |
| Table of hyperparameters | `scripts/make_tables.py` (from `configs/train.yaml`) |
| Estimator, pipeline, ablation, weather, significance, cost, edge–cloud tables | `scripts/make_tables.py` + `scripts/make_tables_extra.py` |
| Figures (data, budgets, trace, trade-off, calibration, scaling, edge–cloud) | `scripts/make_figures.py` |
| Fig. 2 (framework diagram) | `manuscript/figures/src/framework.tex` (TikZ, compiled to `manuscript/figures/framework.pdf`) |

Every number in the manuscript text is a LaTeX macro generated from the result files
(`manuscript/macros/*.tex`), so the text, tables and CSV files cannot drift apart. Result tables report
means over seeds; standard deviations over seeds are in `results/processed/*_std.csv` and seed-level values
in `results/processed/*_seeds.csv`.

## Results at a glance (test period, mean of five seeds)

| Pipeline (K = 4 polls/h of 42 loggers) | Field RMSE (10⁻³ m³ m⁻³) | Onsets detected within 6 h | False alerts/day |
|---|---|---|---|
| Round-robin + last value (usual practice) | 3.39 | 57.1 % | 0.08 |
| Round-robin + VIGIL filter | 3.40 | 62.3 % | 0.20 |
| Rain-KF + VoI index | 3.27 | 59.1 % | 0.05 |
| **VIGIL filter + VoI index** | **3.07** | **66.1 %** | 0.18 |
| Oracle polling (not deployable) | 1.31 | 83.8 % | 0.06 |

All numbers are regenerated by the scripts above; see `manuscript/tables/` and `results/processed/`.

## Licence

Code: MIT (see `LICENSE`). The datasets keep their own licences (see `data/README.md`).
